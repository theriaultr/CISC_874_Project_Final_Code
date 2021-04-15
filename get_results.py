'''
Author: Rachel Theriault (20005337)
The purpose of this file is to contain functions that calculate different metrics and visualize confusion matrices of a network.
The main function to call is get_metrics()
'''

import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def display_confusion_mat(confusion_mat, title_fig, save_name):
  """
  This function displays the confusion matrix using seaborn library
  This is written specifically for this assignment (uses class names etc.)
  It plots a confusion matrix for total and percentage
  Args:
    confusion_mat(2d array of scalars): the confusion matrix for the network with predicted as rows and target as columns
    title_fig(string): the title for the confusion matrix
  """

  #Plot the confusion matrix using total numbers****************************
  #convert 2D array to a dataframe naming the categories
  df_confusion_mat_total = pd.DataFrame(confusion_mat, index = ["Non-IDC", "IDC"], columns = ["Non-IDC", "IDC"])

  #plot the confusion matrix
  plt.figure(figsize = (10,10))
  axis = plt.axes()
  sn.heatmap(df_confusion_mat_total, annot=True, ax=axis)
  plt.ylabel("Output")
  plt.xlabel("Target")
  plt.savefig(save_name+"_Total.png")

  #plot the confusion matrix using overall percentage version***********
  #calculate percentage of each type
  total = np.sum(confusion_mat)
  confusion_mat_percentage = (confusion_mat/total)*100

  #convert to a dataframe
  df_confusion_mat_percentage = pd.DataFrame(confusion_mat_percentage, index = ["Non-IDC", "IDC"], columns = ["Non-IDC", "IDC"])

  #plot the confusion matrix
  plt.figure(figsize = (10,10))
  axis2 = plt.axes()
  sn.heatmap(df_confusion_mat_percentage, annot=True, ax=axis2)
  plt.ylabel("Output")
  plt.xlabel("Target")
  plt.savefig(save_name+"_Percent.png")


def get_metrics(final_prediction, targets, print_statement, figure_name, save_name):
  '''
  The purpose of this function is to calculate the weighted metrics and confusion matrix for a network (precision, recall, F1-score, average)
  Args:
    final_prediction: numpy array of final predictions (probability from each output node)
    targets: numpy array pf targets
    print_statement: what to write in console as print statement before weighted metrics
    figrue_name: The name of the figure
    save_name: The name to use when saving the figures
  '''

  #Get predictions and targets incorrect format for numpy confusion_matrix function
  final_prediction_scalar =  np.zeros(final_prediction.shape[0])
  for i,pred in enumerate(final_prediction):
    final_prediction_scalar[i] = np.argmax(pred)
  
  y_int = np.zeros((final_prediction.shape[0]))
  
  for i,pred in enumerate(targets):
    y_int[i] = int(np.argmax(pred))
  
  #reshape the vectors
  final_prediction_scalar = final_prediction_scalar.reshape(final_prediction.shape[0],1)
  y_int = y_int.reshape(final_prediction.shape[0],1)

  #produce the confusion matrices
  confusion_mat = metrics.confusion_matrix(y_int, final_prediction_scalar) #(y_true, y_pred)
  display_confusion_mat(np.transpose(confusion_mat), figure_name, save_name)

  #get the accuracy, precision, recall and F-score
  results = metrics.classification_report(y_int, final_prediction_scalar, labels=[0, 1]) #(y,true, y_pred)

  #print the metrics
  print(print_statement)
  print(results)


