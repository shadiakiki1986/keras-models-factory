import pandas as pd
import numpy as np
from  keras_models_factory import utils3


"""
simulated data (copy from p5g)
nb_samples = int(1e3)
"""
def ds_1(nb_samples:int, seed:int, lags:list=[1,2]):
  if nb_samples<=0: raise Exception("nb_samples <= 0")

  np.random.seed(seed)
  X1 = pd.Series(np.random.randn(nb_samples))
  X2 = pd.Series(np.random.randn(nb_samples))
  # https://stackoverflow.com/a/20410720/4126114
  X_model = pd.concat({'main': X1, 'lagged 1': X1.shift(lags[0]), 'lagged 2': X1.shift(lags[1]), 'new': X2}, axis=1).dropna()
                       
  X_model['mult'] = X_model.apply(lambda row: row[2]*row[3], axis=1)
  
  
  # Y = X_model.apply(lambda row: 0.25*row[0] + 0.25*row[1] + 0.25*row[2] + 0.25*row[3], axis=1)
  
  Y = X_model.apply(lambda row: 0.2*row['main'] + 0.2*row['lagged 1'] + 0.2*row['lagged 2'] + 0.2*row['new'] + 0.2*row['mult'], axis=1)
  Y = Y.values.reshape((Y.shape[0],1))

  # drop columns in X_model that LSTM is supposed to figure out
  del X_model['lagged 1']
  del X_model['lagged 2']
  del X_model['mult']

  return X_model.values, Y

"""
Copied from keras integration test

https://github.com/fchollet/keras/blob/master/keras/utils/test_utils.py#L13
"""
from keras.utils.test_utils import get_test_data
def ds_2(**kwargs):
  (X_train, Y_train), (X_test, Y_test) = get_test_data(**kwargs)
  X = np.concatenate((X_train, X_test))
  Y = np.concatenate((Y_train, Y_test))
  return X, Y

"""
very long term memory

num_features: this funtion constructs a vertically-stacked version of sequences of the data,
then unstacks to become the "un-strided" dataset for LSTM
Output X will be of length N_train * num_features
Same for Y

Copied from https://philipperemy.github.io/keras-stateful-lstm/
"""
from numpy.random import choice
def ds_3(N_train:int=100, num_features:int=5):
  one_indexes = choice(a=N_train, size=int(N_train / 2), replace=False)

  # stick to -1..+1 for tanh activation in lstm
  X = np.random.uniform(-1, +1, (N_train, num_features))
  X[one_indexes, 0] = 1
  Y = np.zeros((N_train, num_features))
  Y[one_indexes] = 1

  X = X.reshape((N_train*num_features, 1))
  Y = Y.reshape((N_train*num_features, 1))

  return X, Y
