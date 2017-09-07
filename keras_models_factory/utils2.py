# Helps with striding an input matrix

import numpy as np

# https://stackoverflow.com/a/21230438/4126114
# Testing:
#    running_view(np.array([1,2,3,4,5,6,7,8,9,10]),3,0)
#    running_view(np.array([[1,2],[3,4],[5,6],[7,8],[9,10]]),3,0)
def running_view(arr, window, axis=-1):
    """
    return a running view of length 'window' over 'axis'
    the returned array has an extra last dimension, which spans the window
    """
    shape = list(arr.shape)
    shape[axis] -= (window-1)
    if(shape[axis]==0): raise Exception("Empty data used")
    return np.lib.index_tricks.as_strided(
        arr,
        shape + [window],
        arr.strides + (arr.strides[axis],))

def _load_data_strides(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """
    docX = running_view(data, n_prev, 0)
    docX = np.array([y.T for y in docX])
    return docX

def train_test_split(df, test_size=0.1, look_back=100):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    #X_train, y_train = _load_data(df.iloc[0:ntrn], y.iloc[0:ntrn])
    #X_test, y_test = _load_data(df.iloc[ntrn:], y.iloc[ntrn:])
    # alternative to the for loop in the original load data
    # Note that both the original load data and the stride consume a lot of memory
    X_train = _load_data_strides(df[:ntrn,:], look_back)
    X_test = _load_data_strides(df[ntrn:,:], look_back)

    return (X_train), (X_test)

#---------------------
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.layers import RepeatVector, TimeDistributed, Input
from keras.layers.advanced_activations import LeakyReLU #, PReLU

# https://cmsdk.com/python/lstm--learn-a-simple-lag-in-the-data.html
def build_lstm_vanilla(in_neurons:int, out_neurons:int, lstm_dim:int, enc_dim:int=None):
  model = Sequential()
  model.add(LSTM(lstm_dim,
    return_sequences=False,
    input_shape=(None, in_neurons),
    stateful=False,
    activation='tanh'
    ))
  # can use "dropout" parameter of LSTM(...) constructor instead of the below
  # model.add(Dropout(0.25))

  # DOESNT WORK as demonstrated in test-ml/t5-lstm/p5c
  if enc_dim is not None:
    model.add(Dense(enc_dim, activation='tanh'))

  model.add(Dense(out_neurons, activation='linear'))

  model.compile(loss="mean_squared_error", optimizer="rmsprop")

  model.summary()
  return model

# set below by judging by the crescents of the sin and cos in data generation
# look back = 50 with hidden neurons = 25 => MSE = 0.008
# look back = 100 with hidden neurons = 25 => MSE = 0.05
# ditto with enc_dim = 3 => MSE = ?
# BUT PERHAPS MEASURING WITH MSE LIKE THIS IS NOT GOOD
# BECAUSE I SEE THAT THE PREDICTED SIGNAL IS LAGGED
# I PROBABLY NEED SOME ALIGNMENT BEFORE CALCULATING MSE
#
# Edit: the predicted signal is also a cleaned version (without the noise)
#       The MSE in this case should be computed wrt the clean signal?
#
from keras.initializers import Identity
def build_lstm_ae(in_neurons:int, lstm_dim:int, look_back:int, enc_dim:list=None, optimizer='nadam', out_neurons:int=None):
  if out_neurons is None: out_neurons=in_neurons

  model = Sequential()

  # combined from Simple_LSTM_keras_2 and LSTM book, chap 9, seq2seq
  model.add(LSTM(
    lstm_dim, return_sequences=False if enc_dim is None else True, input_shape=(None, in_neurons), activation='tanh'#,
    #dropout=0.25
  ))

  # stacked LSTM for depth
  # Ref: https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras
  #      https://keras.io/getting-started/sequential-model-guide/
  if enc_dim is not None:
    for i,d in enumerate(enc_dim):
      model.add(LSTM(
        d,
        return_sequences=False if (i+1)==len(enc_dim) else True,
        activation='tanh'#,
        # https://stackoverflow.com/questions/40708169/how-to-initialize-biases-in-a-keras-model
        #bias_initializer='zeros',
        #kernel_initializer = Identity()
      ))

  # Sequence-to-sequence autoencoder
  # https://blog.keras.io/building-autoencoders-in-keras.html
  model.add(RepeatVector(look_back))
  model.add(LSTM(lstm_dim, return_sequences=True, activation='tanh'))

  # not sure where I got this from, but it allows to get multiple features with lags between them
  model.add(TimeDistributed(Dense(out_neurons, activation='linear')))

  model.compile(loss="mean_squared_error", optimizer=optimizer)

  model.summary()
  return model
