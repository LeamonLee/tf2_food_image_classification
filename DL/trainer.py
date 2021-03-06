from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
import hypertune

import os
import numpy as np
import argparse
import shutil
from datetime import datetime

from dataHandler import download_data_to_local_directory, upload_data_to_bucket

print("Tensorflow is running on following devices : ")
print(device_lib.list_local_devices())

isOnGCP = True

def build_model(nbr_classes):

    base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(229, 229, 3)))

    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(nbr_classes, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def build_data_pipelines(batch_size, train_data_path, val_data_path, eval_data_path):

    train_augmentor = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range=25,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_augmentor = ImageDataGenerator(
        rescale = 1. / 255
    )

    train_generator = train_augmentor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(229,229),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(229,229),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    eval_generator = val_augmentor.flow_from_directory(
        eval_data_path,
        class_mode="categorical",
        target_size=(229,229),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, eval_generator

def get_number_of_imgs_inside_folder(directory):

    totalcount = 0

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in [".png", ".jpg", ".jpeg"]:
                totalcount = totalcount + 1
    return totalcount

def train(path_to_data, batch_size, epochs, learning_rate, models_bucket_name=""):

    global isOnGCP

    path_train_data = os.path.join(path_to_data, 'training')
    path_val_data = os.path.join(path_to_data, 'validation')
    path_eval_data = os.path.join(path_to_data, 'evaluation')

    total_train_imgs = get_number_of_imgs_inside_folder(path_train_data)
    total_val_imgs = get_number_of_imgs_inside_folder(path_val_data)
    total_eval_imgs = get_number_of_imgs_inside_folder(path_eval_data)
    print(total_train_imgs, total_val_imgs, total_eval_imgs)
    
    train_generator, val_generator, eval_generator = build_data_pipelines(
        batch_size=batch_size,
        train_data_path=path_train_data,
        val_data_path=path_val_data,
        eval_data_path=path_eval_data
    )

    classes_dict = train_generator.class_indices

    model = build_model(nbr_classes=len(classes_dict.keys()))

    optimizer = Adam(lr=learning_rate) # le-5

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    path_to_save_model = './output'
    if not os.path.isdir(path_to_save_model):
        os.makedirs(path_to_save_model)

    ckpt_saver = ModelCheckpoint(
        path_to_save_model,
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )
    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train_imgs // batch_size,
        validation_data=val_generator,
        validation_steps=total_val_imgs // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, ckpt_saver]
    )

    print("[INFO] Evaluation phase...")

    predictions = model.predict_generator(eval_generator)
    predictions_idxs = np.argmax(predictions, axis=1)

    my_classification_report = classification_report(eval_generator.classes, predictions_idxs, 
                                                        target_names=eval_generator.class_indices.keys())

    my_confusion_matrix = confusion_matrix(eval_generator.classes, predictions_idxs)

    print("[INFO] Classification report : ")
    print(my_classification_report)

    print("[INFO] Confusion matrix : ")
    print(my_confusion_matrix)

    print("Starting evaluation using model.evaluate_generator")
    scores = model.evaluate_generator(eval_generator)
    print("Done evaluating!")
    loss = scores[0]
    print(f"loss for hyptertune = {loss}")
    
    if isOnGCP:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        zipped_folder_name = f'trained_model_{now}_loss_{loss}'
        shutil.make_archive(zipped_folder_name, 'zip', '/usr/src/app/output')
        path_zipped_folder = '/usr/src/app/' + zipped_folder_name + '.zip'
        upload_data_to_bucket(models_bucket_name, path_zipped_folder, zipped_folder_name)
    
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='loss', 
                                            metric_value=loss, global_step=epochs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", type=str, help="Bucket name on google cloud storage",
                        default="")
    parser.add_argument("--models_bucket_name", type=str, help="Bucket name on google cloud storage for saving trained models",
                        default="image-classification-food-model-bucket")
    parser.add_argument("--batch_size", type=int, help="Batch size used by the deep learning model", 
                        default=2)
    parser.add_argument("--epochs", type=int, help="Epochs used by the deep learning model", 
                        default=20)
    parser.add_argument("--learning_rate", type=float, help="Batch size used by the deep learning model", 
                        default=1e-5)
    parser.add_argument('--isOnGCP', help='Is it running on GCP or Local Machine, default: False',
						action='store_true')
    parser.add_argument('--isDownloadDataset', help='Do you need to download dataset from GCP bucket, default: False',
						action='store_true')
    parser.add_argument('--isDummy', help='Do you want to run training with dummy dataset, default: False',
						action='store_true')
    args = parser.parse_args()

    pathDataSet = "./dataset"
    bucketName = "image-recognition-food-data-bucket"
    if args.isDummy:
        path_to_data = pathDataSet + '/dummy'
        bucketName = "image-recognition-dummy-data-bucket"
    else:
        path_to_data = pathDataSet

    if args.bucket_name:
      bucketName = args.bucket_name

    if args.isDownloadDataset:
        print("Downloading of data started ...")
        download_data_to_local_directory(bucketName, pathDataSet)
        print("Download finished!")

    isOnGCP = args.isOnGCP

    if isOnGCP:
        train(path_to_data, args.batch_size, args.epochs, args.learning_rate, args.models_bucket_name)
    else:
        train(path_to_data, args.batch_size, args.epochs, args.learning_rate)
    
    print("Training Finished!")