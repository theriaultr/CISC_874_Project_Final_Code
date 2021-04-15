'''
Author: Rachel Theriault (20005337)
The purpose of this script is to develop the dataset train/test split and save the data in files for using while training models
The data is divided so that each file belongs to only train or test
The function is currently set-up to load all of the data (with different save file naems so can continue experiments)
'''


from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random


def get_patient_ids():
  '''
  The purpose of this function is to produce an array of patient IDs loaded in from the file 'patient_ids.txt'
  Returns:
    patient_ids: list of patient file names
  '''
	#opent the text file, and pull out all entries
	with open("/Users/15rlt/Documents/GitHub//CISC-874-Project/Code/patient_ids.txt") as textFile:
	    lines = [line.split() for line in textFile] #split each row in text file into an array entry
	patient_ids_lists = lines
	patient_ids = []

  #split each row into separate entries of final array
	for entry in patient_ids_lists:
		for patient in entry:
			patient_ids.append(patient)

  #convert to a numpy array
	patient_ids = np.array(patient_ids)
	return patient_ids

def load_data(folder):
  '''
  The purpose of this function is to load all of the images from a single folder into a 4D array
  Args:
    folder: the path to the folder containing all of the images
  Returns:
    images: the images from the file in a 4D numpy array
    labels: the label (0 - non-IDC / 1 - IDC) for each image
  '''

  #initialize arrays
  images = []
  labels = []

  #for each image in the folder
  for file in os.listdir(folder):
    #load the image
    image = Image.open(os.path.join(folder, file)).resize((50,50))
    #get the label from the file name (fifth last character)
    label = file[-5]

    #convert the image to array format
    arr = np.array(image)

    #append the image and label to the corresponding array
    images.append(arr)
    labels.append(label)

  return (np.array(images), np.array(labels) )


#load in the data from all folders
def load_all_data(list_of_addresses):
  '''
  The purpose of this function is to load the images from all files and concatenate them into a single array
  Args:
    list_of_addresses: a list of strings of file locations
  Returns:
    images: a 4D numpy array of images from all files (not shuffled)
    labels: a 1D numpy array of all labels (0 - non-IDC / 1 - IDC) corresponding to each image in images
  '''
  #initialize arrays
  all_images = []
  all_labels = np.array([])
  count = 1

  #for each folder
  for folder in list_of_addresses:
    #load the data from the lable 1 folder and the label 0 folder
    print("Starting New Folder")
    print(folder)
    print("Class 1")
    folder_results_positive = load_data(folder+"/1")
    print("Class 0")
    folder_results_negative = load_data(folder+"/0")

    #concatenate the posiitve and negative labelled images to the all_images and all_labels arrays
    if count == 1:
      all_images = np.concatenate((folder_results_positive[0], folder_results_negative[0]))
      all_labels = np.concatenate((folder_results_positive[1], folder_results_negative[1]))
    else:
      all_images = np.concatenate((all_images, folder_results_positive[0]))
      all_labels = np.concatenate((all_labels, folder_results_positive[1]))
      all_images = np.concatenate((all_images, folder_results_negative[0]))
      all_labels = np.concatenate((all_labels, folder_results_negative[1]))

    count+=1
  return all_images, np.array(all_labels)

'''
Load the data
'''

print("-----------------Accessing data------------------")
#get the list of folders
patient_ids = get_patient_ids()
print(patient_ids.shape)

#add the image path to the id names
file_names = ["../Data/IDC_Data/"+i for i in patient_ids]

#shuffle the file names then split so doesn't see same patient twice (80/20 split):
print("num files:", len(file_names))
random.shuffle(file_names)
num_patients = len(file_names)
num_train = int(np.round(num_patients*0.8))

#load the training and testing data from the associated files
print("num_train:", num_train)
print("--------Loading Training Data----------")
all_data_train = load_all_data(file_names[0:num_train])
X_train = all_data_train[0]
y_train_one = all_data_train[1]
print("Size of X_train:", X_train.shape)

print('--------------Loading Testing Data-------')
all_data_test = load_all_data(file_names[-(num_patients-num_train):])
X_test = all_data_test[0]
print("Size of X_test:", X_test.shape)
y_test_one = all_data_test[1]

#Convert labels to correct format (2D binary array)
y_train = np.zeros((y_train_one.shape[0], 2))
for idx, x in enumerate(y_train_one):
  y_train[idx, int(x)] = 1
print("TRAINING DATA")
print(y_train.shape)

y_test = np.zeros((y_test_one.shape[0], 2))
for idx, x in enumerate(y_test_one):
  y_test[idx, int(x)] = 1

print("TEST DATA")
print(y_test.shape)

print("Total number of samples:", y_train.shape[0] + y_test.shape[0])

print('----------------------Saving the data----------------------')
#Save the data
np.save("../Data/IDC_Data/Split/X_train_patient_new", X_train)
np.save("../Data/IDC_Data/Split/y_train_patient_new", y_train)
np.save("../Data/IDC_Data/Split/X_test_patient_new", X_test)
np.save("../Data/IDC_Data/Split/y_test_patient_new", y_test)




