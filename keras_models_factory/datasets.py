import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from  keras_models_factory import utils3


# simulated data (copy from p5g)
# nb_samples = int(1e3)
def ds_1(nb_samples:int, look_back:int, seed:int):
  if nb_samples<=0: raise Exception("nb_samples <= 0")

  lags = [1, 2]

  if look_back < max(lags):
    raise Exception("Not enough look back provided")

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

  # for lstm, need to stride
  X_calib = utils3._load_data_strides(X_model.values, look_back)
  Y_calib = Y[(look_back-1):]

  # split train/test
  Xc_train, Xc_test, Yc_train, Yc_test = train_test_split(X_calib, Y_calib, train_size=0.8, shuffle=False)

  # print(X_calib.shape, Y_calib.shape, Xc_train.shape, Xc_test.shape, Yc_train.shape, Yc_test.shape)
  # (994, 5, 2) (994, 1) (795, 5, 2) (199, 5, 2) (795, 1) (199, 1)

  return (Xc_train, Yc_train), (Xc_test, Yc_test)

"""
https://github.com/fchollet/keras/blob/master/keras/utils/test_utils.py#L13
"""
from keras.utils.test_utils import get_test_data
def ds_2(**kwargs):
  return get_test_data(**kwargs)

"""
data file downloaded from

- https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
- https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line

and cached in my s3 bucket
"""
import requests
import requests_cache
requests_cache.install_cache()
import csv
import io
from sklearn.preprocessing import MinMaxScaler

def ds_3(look_back:int):
  # Pandas read_csv from url
  # http://stackoverflow.com/questions/32400867/ddg#32400969
  url = "https://s3-us-west-2.amazonaws.com/keras-models-factory/international-airline-passengers.csv"
  with requests.Session() as s:
    download = s.get(url)
    decoded_content = download.content.decode('utf-8')
    df=pd.read_csv(io.StringIO(decoded_content), engine='python', skipfooter=3) # usecols=[1], 
    df.columns.values[1]='passengers'
    # https://stackoverflow.com/questions/43759921/pandas-rename-column-by-position#comment79272163_43759994
    df=df.rename({})

    # 
    dataset = df['passengers'].values.reshape((df.shape[0],1))

    # scale to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # target
    Y = dataset

    # stride
    X_calib = utils3._load_data_strides(dataset, look_back)
    Y_calib = Y[(look_back-1):]

    # split train/test
    Xc_train, Xc_test, Yc_train, Yc_test = train_test_split(X_calib, Y_calib, train_size=0.67, shuffle=False)

    return (Xc_train, Yc_train), (Xc_test, Yc_test)
