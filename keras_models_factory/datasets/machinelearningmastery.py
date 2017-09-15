import requests
import requests_cache
requests_cache.install_cache()
import csv
import io
from sklearn.preprocessing import MinMaxScaler
"""
data file downloaded from

- Dr. Jason Brownlee: Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras
- https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
- https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line

and cached in my s3 bucket
"""
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

    return dataset, dataset

