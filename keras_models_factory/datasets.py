import pandas as pd
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

  return X_model.values, Y

"""
https://github.com/fchollet/keras/blob/master/keras/utils/test_utils.py#L13
"""
from keras.utils.test_utils import get_test_data
def ds_2(**kwargs):
  (X_train, Y_train), (X_test, Y_test) = get_test_data(**kwargs)
  X = np.concatenate((X_train, X_test))
  Y = np.concatenate((Y_train, Y_test))
  return X, Y

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

    return dataset, None

