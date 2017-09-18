"""
data files downloaded from

- Dr. Jason Brownlee series
- https://machinelearningmastery.com
"""

import pandas as pd
import requests
import requests_cache
requests_cache.install_cache()
import csv
import io
from sklearn.preprocessing import MinMaxScaler

def read_s3_datamarket_series_csv(filename:str, column1_rename:str):
  # Pandas read_csv from url
  # http://stackoverflow.com/questions/32400867/ddg#32400969
  url = "https://s3-us-west-2.amazonaws.com/keras-models-factory/"+filename
  with requests.Session() as s:
    download = s.get(url)
    decoded_content = download.content.decode('utf-8')
    df=pd.read_csv(io.StringIO(decoded_content), engine='python', skipfooter=3) # usecols=[1], 

    # https://stackoverflow.com/questions/43759921/pandas-rename-column-by-position#comment79272163_43759994
    df.columns.values[1]=column1_rename
    df=df.rename({})

    dataset = df[column1_rename].values.reshape((df.shape[0],1))

    return dataset

"""
data file downloaded from

- Dr. Jason Brownlee 2016-07-21
- Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras
- https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
- https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line

and cached in my s3 bucket
"""
def ds_3():
  df = read_s3_datamarket_series_csv("international-airline-passengers.csv", 'passengers')

  # scale to 0-1
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(df)

  return dataset, dataset


"""
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

This is just 36 points
Monthly sales over 3 years
"""
def ds_4(do_diff:bool):
  dataset = read_s3_datamarket_series_csv('sales-of-shampoo-over-a-three-ye.csv', 'sales')
  if do_diff:
    dataset = pd.Series(dataset.reshape((dataset.shape[0]))).diff().values
    dataset = dataset.reshape((dataset.shape[0],1))
    # skip first nan coming from diff
    dataset = dataset[1:]
  return dataset, dataset
