import glob
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import storage


path_to_credentials = './credentials/imgae-recognition-tf2-a0df9ae0d52b.json' # with storage admin role
food_classes = ["bread", "dairy_product", "dessert", "egg", "fried_food", "meat",
                "noodles_pasta", "rice", "seafood", "soup", "vegetable"]

def split_data_into_class_folders(path_to_data, class_id):
  imgs_paths = glob.glob(path_to_data + "\\" + "*.jpg")

  for path in imgs_paths:
    basename = os.path.basename(path)

    if basename.startswith(str(class_id) + '_'):
      path_to_save = os.path.join(path_to_data, food_classes[class_id])

      if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
      
      shutil.move(path, path_to_save)

def visualize_images(path_to_data):
  imgs_paths = []
  labels = []
  for r, d, f in os.walk(path_to_data):
    for file in f:
      if file.endswith(".jpg"):
        imgs_paths.append(os.path.join(r, file))
        labels.append(os.path.basename(r))
  fig = plt.figure()

  for i in range(16):
    chosen_index = random.randint(0, len(imgs_paths)-1)
    chosen_img = imgs_paths[chosen_index]
    chosen_label = labels[chosen_index]

    ax = fig.add_subplot(4,4, i+1)
    ax.title.set_text(chosen_label)
    ax.imshow(Image.open(chosen_img))
  
  fig.tight_layout(pad=0.05)
  plt.show()

def get_images_size(path_to_data):

  # imgs_paths = []
  widths = []
  heights = []

  for r, d, f in os.walk(path_to_data):
    for file in f:
      if file.endswith(".jpg"):
        img = Image.open(os.path.join(r, file))
        widths.append(img.size[0])
        heights.append(img.size[1])
        img.close()

  mean_width = sum(widths) / len(widths)
  mean_height = sum(heights) / len(heights)
  median_width = np.median(widths)
  median_height = np.median(heights)

  return mean_width, mean_height, median_width, median_height

def list_blobs(bucket_name):
  storage_client = storage.Client.from_service_account_json(path_to_credentials)
  blobs = storage_client.list_blobs(bucket_name)

  return blobs


def download_data_to_local_directory(bucket_name, local_directory):
  storage_client = storage.Client.from_service_account_json(path_to_credentials)
  blobs = storage_client.list_blobs(bucket_name)

  if not os.path.isdir(local_directory):
      os.makedirs(local_directory)

  for blob in blobs:
      joined_path = os.path.join(local_directory, blob.name)
      if os.path.basename(joined_path) == '':
          if not os.path.isdir(joined_path):
              os.makedirs(joined_path)
      else:
          if not os.path.isfile(joined_path):
              if not os.path.isdir(os.path.dirname(joined_path)):
                  os.makedirs(os.path.dirname(joined_path))
                  
              blob.download_to_filename(joined_path)

def upload_data_to_bucket(bucket_name, path_to_data, bucket_blob_name):
  storage_client = storage.Client.from_service_account_json(path_to_credentials)
  bucket = storage_client.get_bucket(bucket_name)

  blob = bucket.blob(bucket_blob_name)
  blob.upload_from_filename(path_to_data)


if __name__ == "__main__":

  isSplitDataSwitch = False
  isVisualizeData = False
  isPrintDataInsights = False
  isListBlobs = False
  isDownloadData = True

  # pathTrainData = r"C:\Users\livin\Desktop\portfolio\AI\Image_Recognition\dataset_food11\training"
  # pathValData = r"C:\Users\livin\Desktop\portfolio\AI\Image_Recognition\dataset_food11\validation"
  # pathEvalData = r"C:\Users\livin\Desktop\portfolio\AI\Image_Recognition\dataset_food11\evaluation"

  pathTrainData = r"D:\advantech\Leamon\code_snippet\Python\tf2_food_image_classification\dataset\training"
  pathValData = r"D:\advantech\Leamon\code_snippet\Python\tf2_food_image_classification\dataset\validation"
  pathEvalData = r"D:\advantech\Leamon\code_snippet\Python\tf2_food_image_classification\dataset\evaluation"

  dummyBucketName = "image-recognition-dummy-data-bucket"

  if isSplitDataSwitch:
    for i in range(len(food_classes)):
      split_data_into_class_folders(pathTrainData, i)
      split_data_into_class_folders(pathValData, i)
      split_data_into_class_folders(pathEvalData, i)
  
  if isVisualizeData:
    visualize_images(pathTrainData)

  if isPrintDataInsights:
    mean_width, mean_height, median_width, median_height = get_images_size(pathTrainData)
    print("mean_width: ", mean_width)
    print("mean_height: ", mean_height)
    print("median_width: ", median_width)
    print("median_height: ", median_height)

  if isListBlobs:
        blobs = list_blobs(dummyBucketName)

        for blob in blobs:
            print(blob.name)

  if isDownloadData:
      download_data_to_local_directory(dummyBucketName, "./dataset-dummy")