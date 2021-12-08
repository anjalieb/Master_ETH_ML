# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 08:55:29 2021

@author: anjal
"""

###########################################################################################
# Libraries
###########################################################################################
import pandas as pd
import os
from tensorflow import keras
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.utils import np_utils
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import numpy as np
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import seaborn as sns

import json



import argparse
from sklearn.utils import class_weight
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time
from pathlib import Path
from numpy.random import seed

# define the number of patches to be loaded for each ID

# load the list IDs as np array

Y        = pd.read_csv(r'E:\labrot_shared\data\TCIA_LGG_cases_115.csv')
# Y        = pd.read_csv(r'C:\Users\anjal\Documents\Master_ETH\HS21\Projekt\Daten\TCIA_LGG_cases_115.csv')
list_IDs = Y['Filename']
arr_list_IDs = np.array(list_IDs)

# load all labels in  a dict to access the labels via ID
df_l = Y[['Filename', '1p/19q']]
all_lab = {}
for k in range(len(df_l['1p/19q'])):
    if df_l['1p/19q'][k]== 'n/n':
        df_l['1p/19q'][k]= 0
    else:
        df_l['1p/19q'][k]= 1
    
    
for i in range(len(df_l['Filename'])):
    all_lab[df_l['Filename'][i]] = df_l['1p/19q'][i]


class DataGenerator(keras.utils.Sequence):
    # In part adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 20200630

    def __init__(self, arr_list_IDs, all_lab, batch_size=50, dim=(30,30), n_channels=1,
                  n_classes=2, shuffle=True, datadir = r'\Data\Patches_single'):

        'Initialization'
        
        self.dim        = dim
        self.batch_size = batch_size
        self.all_lab    = all_lab
        self.list_IDs   = list(arr_list_IDs)
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.on_epoch_end()
        self.datadir    = datadir
        
       

    def on_epoch_end(self):

      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = np.empty((self.batch_size), dtype=int)
      
      for i, ID in enumerate(list_IDs_temp): # list_IDs_temp wird ersetzt durch die neue partition (automatisch, indem in run_models linie 368 geändert wird)
          
          n_datadir = os.path.join(self.datadir + '/' + ID[0] + '/')
          
          X[i,] = np.expand_dims(np.load(os.path.join(n_datadir + ('patch_%s' %ID[1]) + '.npy'),allow_pickle=True),axis=2)
              
          # Store class
          y[i] = all_lab[ID[0]]


      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
      'Generate one batch of data'

      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
      
      # Find list of IDs
      try:
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
      except:
        print(indexes)

      # Generate data
      X, y = self.__data_generation(list_IDs_temp)

      return X, y

    def __getY__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        y = self.__data_generationY(list_IDs_temp)

        return y

    def __data_generationY(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)


      # Initialization
      y = np.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):

          # Store class
          y[i] = self.all_lab[ID]

      return keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getall__(self):
      'Generate one batch of data'

      # Find list of IDs
      list_IDs_temp = self.list_IDs

      # Generate data
      X, y = self.__data_generation_all(list_IDs_temp)

      return X, y

    def __data_generation_all(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

    
      X = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
      y = np.empty(len(list_IDs_temp), dtype=int)
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          
          n_datadir = os.path.join(self.datadir + '/' + ID[0] + '/')
              
          X[i,] = np.expand_dims(np.load(os.path.join(n_datadir + ('patch_%s' %ID[1]) + '.npy'),allow_pickle=True),axis=2)
              
          # Store class

          y[i] = all_lab[ID[0]]
              
   
      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __data_generation_Y(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        # Initialization
        y = np.empty((len(list_IDs_temp)), dtype=int)
        print(len(list_IDs_temp))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store class
            y[i] = self.all_lab[ID]

        return keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getall_Y__(self):
      'Generate one batch of data'

      # Find list of IDs
      list_IDs_temp = self.list_IDs

      # Generate data
      y = self.__data_generation_Y(list_IDs_temp)

      return y
  
    
  
    