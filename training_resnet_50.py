'''
Author: Rachel Theriault (20005337)
The purpose of this script is to train ResNet50. This script was used to train both multiple and single FC layer options and edits were made to the sequential model
*note: before running the scrip make sure have created director to save the model
In it's current state, the model is set-up for code demo (DR, using only a portion of the data, not saving the model, and producing figures)
'''

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn import metrics
from keras import backend as K
# from tensorflow.python.keras import backend as K

#for running the model

from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

#Import other files for buliding dataset and getting visual results
from get_results import *

'''
This script runs ResNet 50 V2
To run make sure to run activate python_gpu_conda to activate virtual environment developed to run code on GPU
Make sure "build_dataset.py" has been run as well if running newer data
'''

#create function to normalize data
def normalize_data(augmented_image):
  re_scaled_image = (augmented_image/127.5) - 1
  return re_scaled_image


#load the data
print("------------------------Loading the data--------------------------")
X_train = np.load("../Data/IDC_Data/Split/X_train_patient.npy")
y_train = np.load("../Data/IDC_Data/Split/y_train_patient.npy")
X_test = np.load("../Data/IDC_Data/Split/X_test_patient.npy")
y_test = np.load("../Data/IDC_Data/Split/y_test_patient.npy")

#testing purposes run with only first 5000 training samples and 1000 testing samples
# X_train = X_train[0:5000]
# y_train = y_train[0:5000]
# X_test = X_test[0:1000]
# y_test = y_test[0:1000]

print("Number of training samples:", y_train.shape[0])
print("Number of testing samples:", y_test.shape[0])
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


"""Import and Set-Up the Model"""
print("-------------------Downloading ResNet50V2 and building model-----------------")
#Download weights from pre-trained image-net data
model_resnet50 = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(50,50,3)
)

#Build the full model defining input tensor and layers after ResNet
model = Sequential(
    [
     Input(shape=(50,50,3)),
     model_resnet50,
     Flatten(),
     Dropout(0.3),
     Dense(2048, activation='tanh'), #relu because that is the activation function used by ResNet
     Dropout(0.3),
     Dense(2, activation='softmax')
    ]
)

print("Final model summmary:")
print(model.summary())

"""Data augmentation"""
#How to perform real-time data augmentation was accessed by https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
#Create data augmentation object
aug = ImageDataGenerator(
    rotation_range=30,
    brightness_range = (0.5, 1.5),
		horizontal_flip=True,
    vertical_flip = True,
		fill_mode="reflect",
    preprocessing_function = normalize_data)

"""Compile the model"""
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate = 0.001), #function used to minimize the loss (back propogation and gradient descent)
    loss=tf.keras.losses.BinaryCrossentropy(), #defines how far off a prediction is from correct answer - loss is high if model predicts small prob classified correctly
    metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalAccuracy()]
)

"""Train Model 1 --> only last layers"""

print("-----------------------TRAINING ADDED LAYERS ONLY - LR 1e-3----------------------")
#Set ResNet trainable to False
print("Model layers: ", model.layers)
model.layers[0].trainable = False

#fit the model using real-time data augmentation with batch size 32 and early stoppping to prevent overfitting
history1 = model.fit(
  x = aug.flow(X_train, y_train, batch_size = 32),
  shuffle=True,
  epochs = 200,
  callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta = 0.001, restore_best_weights=True)])
  

#save the model
print("---------------------------- Saving Model ---------------------------------")
model.save("Saved_Models/Train1")

"""Plot the results of the training session 1"""
print("---------------------------------- Getting Metrics -----------------------------")
plt.figure(figsize=(10,10))
plt.plot(history1.history['mean_squared_error'])
plt.title('MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.savefig('Train_1_MSE_100')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(history1.history['categorical_accuracy'])
plt.title('Categorical Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('Train_1_Categorical_Accuracy_100')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(history1.history['loss'])
plt.title('Binary Cross Entropy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('Train_1_Loss_100')
plt.close()

#Debugging:
# print("Type of X_train:", type(X_train))
# print("data type entry:", type(X_train[0,0,0,0]))

#normalize all samples for evaluation with non-augmented training and testing data
X_train_normalized = np.zeros(X_train.shape)

for idx, sample in enumerate(X_train):
  X_train_normalized[idx] = normalize_data(sample)
  if idx==0:
    print("size of a sample normalizing:", sample.shape)

X_test_normalized = np.zeros(X_test.shape)
for idx, sample in enumerate(X_test):
  X_test_normalized[idx] = normalize_data(sample)

#predict for training and testing data
final_prediction_train = model.predict(X_train_normalized.astype(np.float64))
final_prediction_test = model.predict(X_test_normalized.astype(np.float64))

#get the metrics (precision, recall, F-Score, accuracy, confusion matrix) fro train and test data
get_metrics(final_prediction_train, y_train, "Train Metrics***************", "Train_1_100", "Train_1_Conf_Mat_100")
get_metrics(final_prediction_test, y_test, "Test Metrics****************", "Test_1_100", "Test_1_Conf_Mat_100")



# """Train model part 2 --> training whole model with lower learning rate"""

# print("-----------------------TRAINING ALL LAYERS - LR 1e-6----------------------")

# K.set_value(model.optimizer.learning_rate, 0.000001)
# model.trainable = True
# # model.summary()
# # print(len(model.layers))
# # print(model.layers[0].trainable)
# # print(model.layers[1].trainable)
# # print(model.layers[2].trainable)
# # print(model.layers[3].trainable)
# # print(model.layers[4].trainable)


# #    validation_data = augval.flow(X_valid, y_valid, batch_size=128),
# history2 = model.fit(
#     x = aug.flow(X_train, y_train, batch_size = 128),
#     shuffle = True,
#     epochs = 300,
#     callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta = 0.0001, restore_best_weights=True)]
# )

# print("------ saving model ----------")
# model.save("Saved_Models/Train2")

# #Calculate results from training the full model

# print("------- getting metrics ------")
# """Plot the results of the training session 12"""
# plt.figure(figsize=(10,10))
# plt.plot(history2.history['mean_squared_error'])
# # plt.plot(history2.history['val_mean_squared_error'])
# plt.title('MSE Fine-Tuning Whole Model')
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# # plt.legend(["Train", "Validation"])
# plt.savefig('Train_2_MSE')
# plt.close()

# plt.figure(figsize=(10,10))
# plt.plot(history2.history['categorical_accuracy'])
# # plt.plot(history2.history['val_categorical_accuracy'])
# plt.title('Categorical Accuracy Fine-Tuning Whole Model')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# # plt.legend(["Train", "Validation"])
# plt.savefig('Train_2_Categorical_Accuracy')
# plt.close()

# plt.figure(figsize=(10,10))
# plt.plot(history2.history['loss'])
# # plt.plot(history2.history['val_loss'])
# plt.title('Categorical Cross Entropy Fine-Tuning Whole Model')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# # plt.legend(["Train", "Validation"])
# plt.savefig('Train_2_Loss')
# plt.close()

# #Get training, validation and testing results
# final_prediction_train = model.predict(X_train)
# # final_prediction_valid = model.predict(X_valid)
# final_prediction_test = model.predict(X_test)

# get_metrics(final_prediction_train, y_train, "Train Metrics***************", "Train_2", "Train_2_Conf_Mat")
# # get_metrics(final_prediction_valid, y_valid, "VALIDATION Metrics**********", "Valid_2", "Valid_2_Conf_Mat")
# get_metrics(final_prediction_test, y_test, "Test Metrics****************", "Test_2", "Test_2_Conf_Mat")
