# CISC_874_Project_Final_Code

CISC 874 Final Code
Author: Rachel Theriault
Data can be located at: https://www.kaggle.com/paultimothymooney/breast-histopathology-images 

The final models were too large to be stored on GitHub

The goal of the project was to train ResNet50 adn ResNet101 fior IDC versus non-IDC classification and use a variaety of ensemble techniques to enhance sensitivity of IDC detection

Building the Dataset:
- build_dataset.py --> build and save the dataset
- patients.txt --> contains the patient ids matching the files

Running ResNet50 and ResNet 101:
- training_resnet50 --> used to train and save ResNet50
- training_resnet101 --> used to train and save ResNet101

Running Ensemble:
- extension_2_ensemble --> ensemble training for all ensemble options for the second last layer of the model (dimension 2048)
- extension_output_ensemble --> ensemble training for all ensemble options for the output of models (dimension 2)
