'''
The purpose of this section is to extend ResNet in an ensemble approach using the final layer before model output. 
It designed for combining 2 models with the same number of nodes on last fully connected layer before output layer
The optional methods set at the parameter OPTION are 
1. Summing fully connected portion (SUM)
2. Multiplying fully connected portions (MULT)
3. Concatenating fully connected portions (CONC)
4. RNN (RNN)
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
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input, Concatenate, Add, Average, SimpleRNN, Embedding, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential

from get_results import *


def normalize_data(augmented_image):
    '''
    Purpose: scale the data sample-wise between -1 and 1
    Args:
        augmented_image: array of the image to be scaled
    Returns:
        re_scaled_image: augmented_image scaled between -1 and 1
    Assumption: values range from 0 to 255
    '''
    re_scaled_image = (augmented_image/127.5) - 1
    return re_scaled_image

'''
SET THE PARAMETERS
'''
OPTION = "SUM"
NUM_EPOCHS = 50
BATCH_SIZE = 32
TYPE_NAME = "With_Ensemble_DR_LR1e4" #name of the test being performed
ENSEMBLE_DR_RATE = 0.3

'''
LOAD THE MODELS FROM THE PATH NAME SAVED
'''
model1_path = "Saved_Models/Train1_50_No_Dropout/saved_model.pb"
model2_path = "Saved_Models/Train1_100_No_Dropout/saved_model.pb"
final_layer_size = 2048
lr_extension = 0.0001

print("------------------Loading the models------------------------")
model1 = load_model('Saved_Models/Train1_50_DR_Final/')
model2 = load_model('Saved_Models/Train1_100_DR_Final/')
model1._name = "Train1"
model2._name= "Train2"
#remove the last layer of the model
model1.pop()
model2.pop()
#don't re-train the models
model1.trainable=False
model2.trainable=False
print("model 1 layers trainable:")
print(model1.layers[0].trainable)
print(model1.layers[1].trainable)
print(model1.layers[2].trainable)

#re-train from scratch option
# #model 1******************************************************
# model_resnet50 = tf.keras.applications.ResNet50V2(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=(50,50,3)
# )

# model1 = Sequential(
#     [
#      Input(shape=(50,50,3)),
#      model_resnet50,
#      Flatten(),
#      Dense(2048, activation='tanh'), #relu because that is the activation function used by ResNet
#     ]
# )

# print("Model 1 layer 0:", model1.layers[0])
# print("Model 1 layer 1:", model1.layers[1])

# model_resnet101 = tf.keras.applications.ResNet101V2(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=(50,50,3)
# )

# model2 = Sequential(
#     [
#      Input(shape=(50,50,3)),
#      model_resnet101,
#      Flatten(),
#      Dense(2048, activation='tanh'), #relu because that is the activation function used by ResNet
#     ]
# )

'''
Load the training data
'''
#load train data
print("------------------------Loading the data--------------------------")
X_train = np.load("../Data/IDC_Data/Split/X_train_patient.npy")
y_train = np.load("../Data/IDC_Data/Split/y_train_patient.npy")
X_test = np.load("../Data/IDC_Data/Split/X_test_patient.npy")
y_test = np.load("../Data/IDC_Data/Split/y_test_patient.npy")


#Create model for each option and train the model
if OPTION == "SUM":
    print("Using summation option")
    #develop a new model that will sum the outputs of each model before prediction 
    inputs = Input(shape=(50,50,3))
    model1_layer = model1(inputs)
    model2_layer = model2(inputs)
    addition_layer = Add()([model1_layer, model2_layer]) #concatenation layer
    dropout1 = Dropout(ENSEMBLE_DR_RATE)(addition_layer)
    dense1 = Dense(512, activation='relu')(dropout1)
    dropout2 = Dropout(ENSEMBLE_DR_RATE)(dense1)
    # dense1 = Dense(final_layer_size, activation='relu')(addition_layer)
    # dense2 = Dense(final_layer_size/2, activation='relu')(dense1)
    out = Dense(2, activation='softmax')(dropout2)
    model = Model(inputs = inputs, outputs = out)

    print("Summation model summary:")
    model.summary()

    #create the augmenting data object
    aug = ImageDataGenerator(
        rotation_range=30,
        brightness_range = (0.5, 1.5),
		horizontal_flip=True,
        vertical_flip = True,
		fill_mode="reflect",
        preprocessing_function=normalize_data)

    #compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate = lr_extension), #function used to minimize the loss (back propogation and gradient descent)
        loss=tf.keras.losses.BinaryCrossentropy(), #defines how far off a prediction is from correct answer - loss is high if model predicts small prob classified correctly
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalAccuracy()]
    )


    #sending the same augmented input through both models, fit the model to the data
    history1 = model.fit(
      x = aug.flow(X_train, y_train, batch_size = BATCH_SIZE),
      shuffle = True,
      epochs = NUM_EPOCHS,
      callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta = 0.001, restore_best_weights=True)])

elif OPTION == "AVG":
    print("Using average option")
    #develop a new model that will average the outputs of each model before prediction 
    inputs = Input(shape=(50,50,3))
    model1_layer = model1(inputs)
    model2_layer = model2(inputs)
    avg_layer = Average()([model1_layer, model2_layer]) #average layer
    dropout1 = Dropout(ENSEMBLE_DR_RATE)(avg_layer)
    layer1 = Dense(512, activation = 'relu')(dropout1)
    dropout2 = Dropout(ENSEMBLE_DR_RATE)(layer1)
    #additional layers used in testing
    # layer2 = Dense(128, activation = 'tanh')(dropout2)
    # dropout3 = Dropout(ENSEMBLE_DR_RATE)(layer2)
    # layer3 = Dense(32, activation = 'tanh')(dropout3)
    # dropout4 = Dropout(ENSEMBLE_DR_RATE)(layer3)
    # # dense1 = Dense(final_layer_size, activation='relu')(addition_layer)
    # dense2 = Dense(final_layer_size/2, activation='relu')(dense1)
    out = Dense(2, activation='softmax')(dropout2)
    model = Model(inputs = inputs, outputs = out)

    print("Summary of Avergae Model:")
    model.summary()
    #create the augmenting data object
    aug = ImageDataGenerator(
        rotation_range=30,
        brightness_range = (0.5, 1.5),
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode="reflect",
        preprocessing_function=normalize_data)

    #compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate = lr_extension), #function used to minimize the loss (back propogation and gradient descent)
        loss=tf.keras.losses.BinaryCrossentropy(), #defines how far off a prediction is from correct answer - loss is high if model predicts small prob classified correctly
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalAccuracy()]
        )

    model.summary()

    #sending the same augmented input through both models and fit the model
    history1 = model.fit(
        x = aug.flow(X_train, y_train, batch_size = BATCH_SIZE),
        shuffle = True,
        epochs = NUM_EPOCHS,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta = 0.001, restore_best_weights=True)])

    
elif OPTION == "CONCAT":
    print("Using conatenation option")
    #develop a new model that will concatenate the outputs of each model before prediction 
    inputs = Input(shape=(50,50,3))
    model1_layer = model1(inputs)
    model2_layer = model2(inputs)
    concat_layer = Concatenate()([model1_layer, model2_layer]) #concatenation layer
    dropout1 = Dropout(ENSEMBLE_DR_RATE)(concat_layer)
    layer1 = Dense(final_layer_size, activation='relu')(dropout1)
    dropout2 = Dropout(ENSEMBLE_DR_RATE)(layer1)
    layer2 = Dense(512, activation='relu')(dropout2)
    dropout3 = Dropout(ENSEMBLE_DR_RATE)(layer2)
    out = Dense(2, activation='softmax')(dropout3)
    model = Model(inputs = inputs, outputs = out)
    model.summary()

    #create the augmenting data object
    aug = ImageDataGenerator(
        rotation_range=30,
        brightness_range = (0.5, 1.5),
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode="reflect",
        preprocessing_function=normalize_data)

    #compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate = lr_extension), #function used to minimize the loss (back propogation and gradient descent)
        loss=tf.keras.losses.BinaryCrossentropy(), #defines how far off a prediction is from correct answer - loss is high if model predicts small prob classified correctly
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalAccuracy()]
        )

        
    #sending the same augmented input through both models and fit the model
    history1 = model.fit(
        x = aug.flow(X_train, y_train, batch_size = 32),
        shuffle = True,
        epochs = NUM_EPOCHS,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta = 0.001, restore_best_weights=True)])

elif OPTION == "RNN":
    print("Using RNN option")
    #develop a new model that will concatenate the outputs of each model before prediction 
    inputs = Input(shape=(50,50,3))
    model1_layer = model1(inputs)
    model2_layer = model2(inputs)
    new_input = tf.convert_to_tensor([model1_layer, model2_layer])
    new_input = tf.transpose(new_input, [1,0,2]) #shape (None, timestep(2), features(2048))
    simple_rnn = SimpleRNN(2048, activation='relu', dropout=0.3)(new_input) # Shape = (None, 2048)
    dense1 = Dense(512, activation='relu')(simple_rnn)
    dropout1 = Dropout(ENSEMBLE_DR_RATE)(dense1)
    out = Dense(2, activation='softmax')(dropout1)
    model = Model(inputs = inputs, outputs = out)

    model.summary()

    #create the augmenting data object
    aug = ImageDataGenerator(
        rotation_range=30,
        brightness_range = (0.5, 1.5),
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode="reflect",
        preprocessing_function=normalize_data)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate = lr_extension), #function used to minimize the loss (back propogation and gradient descent)
        loss=tf.keras.losses.BinaryCrossentropy(), #defines how far off a prediction is from correct answer - loss is high if model predicts small prob classified correctly
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalAccuracy()])

    model.layers[0].trainable = False

        #sending the same augmented input through both models
    history1 = model.fit(
        x = aug.flow(X_train, y_train, batch_size = BATCH_SIZE),
        shuffle = True,
        epochs = NUM_EPOCHS,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta = 0.001, restore_best_weights=True)])

else:
    print("Option selected is not defined. Please select from SUM, MULT, CONC, or RNN")



"""Plot the results of the training session 1"""
print("---------------------------------- Getting Metrics -----------------------------")
plt.figure(figsize=(10,10))
plt.plot(history1.history['mean_squared_error'])
plt.title('MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.savefig(OPTION+'_Train_MSE_'+TYPE_NAME)
plt.close()

plt.figure(figsize=(10,10))
plt.plot(history1.history['categorical_accuracy'])
plt.title('Categorical Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig(OPTION+'_Train_Categorical_'+TYPE_NAME)
plt.close()

plt.figure(figsize=(10,10))
plt.plot(history1.history['loss'])
plt.title('Binary Cross Entropy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig(OPTION+'_Train_Loss_'+TYPE_NAME)
plt.close()

#Get training and testing results
#normalize all samples
X_train_normalized = np.zeros(X_train.shape)

for idx, sample in enumerate(X_train):
  X_train_normalized[idx] = normalize_data(sample)
  if idx==0:
    print("size of a sample normalizing:", sample.shape)

X_test_normalized = np.zeros(X_test.shape)
for idx, sample in enumerate(X_test):
  X_test_normalized[idx] = normalize_data(sample)


#Perform final prediction of the model using non-augmented train and test data
final_prediction_train = model.predict(X_train_normalized.astype(np.float64))
final_prediction_test = model.predict(X_test_normalized.astype(np.float64))

#get the final metrics (from file get_results)
get_metrics(final_prediction_train, y_train, "Train Metrics***************", "Train", OPTION+"_Ensemble_Confusion_Train_Ensemble_"+TYPE_NAME)
get_metrics(final_prediction_test, y_test, "Test Metrics****************", "Test", OPTION+"_Ensemble_Confusion_Test_Ensemble_"+TYPE_NAME)