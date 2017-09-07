# Deps
# sudo apt-get install python3-tk
# pip install nose
#
# Run all
# nosetests --logging-level INFO test_lstm.py
#
# Run a single test class with unittest
# http://pythontesting.net/framework/specify-test-unittest-nosetests-pytest/
# https://nose.readthedocs.io/en/latest/plugins/logcapture.html
# nosetests --logging-level INFO --nocapture -v test_lstm.py:TestP1Core.test_fit_model_1
# nosetests --logging-level INFO --nocapture -v test_lstm.py:TestP1Core.test_fit_model_2


import unittest
from unittest_data_provider import data_provider
from  keras_models_factory import lstm, utils, utils2, utils3

import numpy as np
import pandas as pd

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K # copy from ~/.local/share/virtualenvs/G2ML/lib/python3.5/site-packages/keras/callbacks.py
from hashlib import md5

from keras.models import load_model
from os import path, makedirs
import nose
import json

# https://stackoverflow.com/a/22721724/4126114
from collections import OrderedDict
def sortOD(od):
  res = OrderedDict()
  for k, v in sorted(od.items()):
    if isinstance(v, dict):
      res[k] = sortOD(v)
    else:
      res[k] = v
  return res

class TestP1Core(object): #unittest.TestCase): # https://stackoverflow.com/questions/6689537/nose-test-generators-inside-class#comment46280717_11093309

  #-------------------------
  # save model in file: filename is md5 checksum of models file
  # This way, any edits in the file result in a new filename and hence re-calculating the model
  def setUp(self):
    self._model_path = path.join("/", "tmp", "test-ml-cache")

  #-------------------------
  # simulated data (copy from p5g)
  # nb_samples = int(1e3)
  def _data(self,nb_samples:int):
    if nb_samples<=0: raise Exception("nb_samples <= 0")
    np.random.seed(0) # https://stackoverflow.com/a/34306306/4126114

    lags = [1, 2]
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

    return (X_model, Y, lags)

  #-------------------------
  # get initial epoch
  def _get_initial_epoch(self,tb_log_dir):
    if not path.exists(tb_log_dir):
        print("tensorboard history not found: %s"%(tb_log_dir))
        return 0

    print("tensorboard history found: %s"%(tb_log_dir))
    latest = utils3.load_tensorboard_latest_data(tb_log_dir)
    if latest is None:
        print("tensorboard history is empty")
        return 0

    initial_epoch = latest['step']+1 # 0-based
    print(
        "found history on trained model: epochs: %i, loss: %s, val_loss: %s" %
        (initial_epoch, latest['loss'], latest['val_loss'])
    )

    return initial_epoch

  #-------------------------
  #  epochs = 300
  #  look_back = 5
  def _fit(self, X_model:pd.DataFrame, Y, lags:list, model, epochs:int, look_back:int, model_file:str, keras_file:str):
    if epochs<=0: raise Exception("epochs <= 0")

    if look_back < max(lags):
        raise Exception("Not enough look back provided")
    X_calib = utils3._load_data_strides(X_model.values, look_back)
    
    Y_calib = Y[(look_back-1):]


    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                               patience=100)
    checkpointer = ModelCheckpoint(filepath=keras_file,
                               verbose=0, #2
                               save_best_only=True)
    # https://stackoverflow.com/a/43549608/4126114
    tb_log_dir = path.join(model_file, 'tb')
    tensorboard = TensorBoard(log_dir=tb_log_dir,
                     histogram_freq=10,
                     write_graph=True,
                     write_images=False)

      
    history = model.fit(
        x=X_calib,
        y=Y_calib,
        epochs = epochs,
        verbose = 0,#2,
        batch_size = 1000, # 100
        validation_split = 0.2,
        callbacks = [early_stopping, checkpointer, tensorboard],
        initial_epoch = self._get_initial_epoch(tb_log_dir),
        shuffle=False
    )
    
    pred = model.predict(x=X_calib, verbose = 0)

    # reset tensorflow session
    # https://stackoverflow.com/questions/43975090/tensorflow-close-session-on-object-destruction
    # found in /home/ubuntu/.local/share/virtualenvs/G2ML/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py 
    K.clear_session()
    
    err = utils.mse(Y_calib, pred)

    return (history, err)

  #-------------------------
  params = (
    (int(10e3),  600, 0.0241, [10]),
    (int(10e3),  600, 0.0147, [10,10]),
    (int(10e3),  600, 0.0173, [10,10,10]),
    (int(10e3),  600, 0.0528, [10,10,10,10]),
    (int(10e3),  600, 0.0093, [30]),
    (int(10e3),  600, 0.0097, [60]),
    (int(10e3),  600, 0.0061, [90]),
#    (int(10e3),  600, 0.0146, [30,10]),
#    (int(10e3),  600, 0.0082, [30,30]),
#    (int(10e3),  600, 0.0085, [30,60]),
#    (int(10e3),  600, 0.0079, [60,30]),
#    (int(10e3),  600, 0.0192, [60,60]),
#    (int(10e3),  600, 0.0054, [90,60]),
#    (int(10e3),  600, 0.0086, [90,60,30]),
#
#    # tests with less epochs
#    (int(10e3),  400, 0.01, [90,60,30]),
#    (int(10e3),  400, 0.01, [30,30]),
#    (int(10e3),  300, 0.0129, [60,30]),
#
#    # failed tests
#    # stuck since epoch 400 # (int(10e3), 1000, 0.01, [30,20,10]),
#
#    # tests with less data
#    (int( 1e3), 3000, 0.01, [30]),
#    (int( 1e3), 2100, 0.01, [60]),
#    (int( 1e3), 4000, 0.01, [30, 20, 10]),

#    # testing tests
#    (int( 1e3), 30, 0.6153, [30]),
#    (int( 1e3), 20, 0.7024, [60]),

  )

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  def test_fit_model_1(self):
    for nb_samples, epochs, expected_mse, lstm_dim in self.params:
      yield self.check_fit_model_1, nb_samples, epochs, expected_mse, lstm_dim

  def check_fit_model_1(self, nb_samples, epochs, expected_mse, lstm_dim):
    model_desc = "model_1: nb %s, epochs %s, dim %s"%(nb_samples, epochs, lstm_dim)
    print(model_desc)
    print("model_1: mse %s"%(expected_mse))

    (X_model, Y, lags) = self._data(nb_samples)

    look_back = 5
    model, model_file, keras_file = self._model(lambda: lstm.model_1(X_model.shape[1], lstm_dim, look_back))

    # model = utils2.build_lstm_ae(X_model.shape[1], lstm_dim[0], look_back, lstm_dim[1:], "adam", 1)
    # model.summary()
    (history, err) = self._fit(X_model, Y, lags, model, epochs, look_back, model_file, keras_file)

    # with 10e3 points
    #      np.linalg.norm of data = 45
    #      and a desired mse <= 0.01
    # The minimum loss required = (45 * 0.01)**2 / 10e3 ~ 2e-5
    #
    # with 1e3 points
    #      np.linalg.norm of data = 14
    #      and a desired mse <= 0.01
    # The minimum loss required = (14 * 0.01)**2 / 1e3 ~ 2e-5 (also)
    nose.tools.assert_almost_equal(err, expected_mse, places=4)

  #--------------------
  # callback: lambda function without any parameters and returning a keras model
  def _model(self,callback):
    model = callback()
    model_file = [sortOD(x) for x in model.get_config()]
    model_file = json.dumps(model_file).encode('utf-8')
    model_file = md5(model_file).hexdigest()
    model_file = path.join(self._model_path, model_file)
    print("model file", model_file)

    # create folders in model_file
    makedirs(model_file, exist_ok=True)

    # proceed
    keras_file = path.join(model_file, 'keras')
    print("keras file", keras_file)
    if not path.exists(keras_file):
      print("launch new model")
      return model, model_file, keras_file

    print("load pre-trained model")
    model = load_model(keras_file)
    # model.summary()
    return model, model_file, keras_file

  #-------------------------
  def test_fit_model_2(self):
    for (nb_samples, epochs, expected_mse, lstm_dim) in self.params:
      yield self.check_fit_model_2, nb_samples, epochs, expected_mse, lstm_dim
      #self.check_fit_model_2( nb_samples, epochs, expected_mse, lstm_dim )

  def check_fit_model_2(self, nb_samples, epochs, expected_mse, lstm_dim):
    model_desc = "model 2: nb %s, epochs %s, mse %s, dim %s"%(nb_samples, epochs, expected_mse, lstm_dim)
    print(model_desc)
    print("model 2: mse %s"%(expected_mse))
    (X_model, Y, lags) = self._data(nb_samples)

    look_back = 5
    model, model_file, keras_file = self._model(lambda: lstm.model_2(X_model.shape[1], lstm_dim, look_back))
    # model.summary()
    (history, err) = self._fit(X_model, Y, lags, model, epochs, look_back, model_file, keras_file)

    nose.tools.assert_almost_equal(err, expected_mse, places=4)
