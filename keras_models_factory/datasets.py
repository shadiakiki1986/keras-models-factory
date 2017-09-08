import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from  keras_models_factory import utils3


# simulated data (copy from p5g)
# nb_samples = int(1e3)
def ds_1(self, nb_samples:int, look_back:int):
  if nb_samples<=0: raise Exception("nb_samples <= 0")
  np.random.seed(0) # https://stackoverflow.com/a/34306306/4126114

  lags = [1, 2]

  if look_back < max(lags):
    raise Exception("Not enough look back provided")

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

  # for lstm, need to stride
  X_calib = utils3._load_data_strides(X_model.values, look_back)
  Y_calib = Y[(look_back-1):]

  # split train/test
  Xc_train, Xc_test = train_test_split(X_calib, train_size=0.8, shuffle=False)
  Yc_train, Yc_test = train_test_split(Y_calib, train_size=0.8, shuffle=False)

  return (Xc_train, Yc_train), (Xc_test, Yc_test)

############################
# https://github.com/fchollet/keras/blob/master/keras/utils/test_utils.py#L13
from keras.utils.test_utils import get_test_data
def ds_2(**kwargs):
  return get_test_data(**kwargs)
