from  keras_models_factory import lstm, utils, utils2, utils3

import numpy as np
import pandas as pd

# copy from ~/.local/share/virtualenvs/G2ML/lib/python3.5/site-packages/keras/callbacks.py
from keras import backend as K

import nose

from test_base import TestBase

class TestLstm(TestBase):

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
  #  epochs = 300
  #  look_back = 5
  def _fit(self, X_model:pd.DataFrame, Y, lags:list, model, epochs:int, look_back:int, model_file:str, keras_file:str):
    if epochs<=0: raise Exception("epochs <= 0")

    if look_back < max(lags):
        raise Exception("Not enough look back provided")
    X_calib = utils3._load_data_strides(X_model.values, look_back)
    
    Y_calib = Y[(look_back-1):]

    tb_log_dir, callbacks = self.get_callbacks(model_file, keras_file)
    history = model.fit(
        x=X_calib,
        y=Y_calib,
        epochs = epochs,
        verbose = 0,#2,
        batch_size = 1000, # 100
        validation_split = 0.2,
        callbacks = callbacks,
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
  params_1 = (
#    # SLOW TEST
#    (int(10e3),  600, 0.0241, [10]),
#    (int(10e3),  600, 0.0147, [10,10]),
#    (int(10e3),  600, 0.0173, [10,10,10]),
#    (int(10e3),  600, 0.0528, [10,10,10,10]),
#    (int(10e3),  600, 0.0093, [30]),
#    (int(10e3),  600, 0.0097, [60]),
#    (int(10e3),  600, 0.0061, [90]),
#    (int(10e3),  600, 0.0146, [30,10]),
#    (int(10e3),  600, 0.0082, [30,30]),
#    (int(10e3),  600, 0.0085, [30,60]),
#    (int(10e3),  600, 0.0079, [60,30]),
#    (int(10e3),  600, 0.0192, [60,60]),
#    (int(10e3),  600, 0.0054, [90,60]),
#    (int(10e3),  600, 0.0086, [90,60,30]),

#    # tests with less epochs
#    (int(10e3),  400, 0.01, [90,60,30]),
#    (int(10e3),  400, 0.01, [30,30]),
#    (int(10e3),  300, 0.0129, [60,30]),

#    # failed tests
#    # stuck since epoch 400 # (int(10e3), 1000, 0.01, [30,20,10]),

    # tests with less data
    (int( 1e3), 3000, 0.0059, [30]),
    (int( 1e3), 2100, 0.0112, [60]),
    (int( 1e3), 4000, 0.0097, [30, 20, 10]),

    # quick tests
    (int( 1e3), 30, 0.6153, [30]),
    (int( 1e3), 20, 0.7024, [60]),

  )

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  def test_fit_model_1(self):
    for nb_samples, epochs, expected_mse, lstm_dim in self.params_1:
      yield self.check_fit_model_1, nb_samples, epochs, expected_mse, lstm_dim

  def check_fit_model_1(self, nb_samples, epochs, expected_mse, lstm_dim):
    train_desc = "model_1: nb %s, epochs %s"%(nb_samples, epochs)
    print(train_desc)
    print("model_1: mse %s, dim %s"%(expected_mse, lstm_dim))

    (X_model, Y, lags) = self._data(nb_samples)

    look_back = 5
    model, model_file, keras_file = self._model(lambda: lstm.model_1(X_model.shape[1], lstm_dim, look_back), train_desc)

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

  #-------------------------
  params_2 = (
#    # SLOW TEST
#    (int(10e3),  600, 0.0241, [10]),
#    (int(10e3),  600, 0.0147, [10,10]),
#    (int(10e3),  600, 0.0173, [10,10,10]),
#    (int(10e3),  600, 0.0528, [10,10,10,10]),
#    (int(10e3),  600, 0.0093, [30]),
#    (int(10e3),  600, 0.0097, [60]),
#    (int(10e3),  600, 0.0061, [90]),
#    (int(10e3),  600, 0.0146, [30,10]),
#    (int(10e3),  600, 0.0082, [30,30]),
#    (int(10e3),  600, 0.0085, [30,60]),
#    (int(10e3),  600, 0.0079, [60,30]),
#    (int(10e3),  600, 0.0192, [60,60]),
#    (int(10e3),  600, 0.0054, [90,60]),
#    (int(10e3),  600, 0.0086, [90,60,30]),

#    # tests with less epochs
#    (int(10e3),  400, 0.01, [90,60,30]),
#    (int(10e3),  400, 0.01, [30,30]),
#    (int(10e3),  300, 0.0129, [60,30]),

#    # failed tests
#    # stuck since epoch 400 # (int(10e3), 1000, 0.01, [30,20,10]),

    # tests with less data
    (int( 1e3), 3000, 0.0058, [30]),
    (int( 1e3), 2100, 0.0110, [60]),
    (int( 1e3), 4000, 0.01, [30, 20, 10]),

    # quick tests
    (int( 1e3), 30, 0.6128, [30]),
    (int( 1e3), 20, 0.0054, [60]),

  )

  #-------------------------
  def test_fit_model_2(self):
    for (nb_samples, epochs, expected_mse, lstm_dim) in self.params_2:
      yield self.check_fit_model_2, nb_samples, epochs, expected_mse, lstm_dim
      #self.check_fit_model_2( nb_samples, epochs, expected_mse, lstm_dim )

  def check_fit_model_2(self, nb_samples, epochs, expected_mse, lstm_dim):
    print("model 2: mse %s, dim %s"%(expected_mse, lstm_dim))
    train_desc = "nb %s, epochs %s"%(nb_samples, epochs)
    print(train_desc)
    (X_model, Y, lags) = self._data(nb_samples)

    look_back = 5
    model, model_file, keras_file = self._model(lambda: lstm.model_2(X_model.shape[1], lstm_dim, look_back), train_desc)
    # model.summary()
    (history, err) = self._fit(X_model, Y, lags, model, epochs, look_back, model_file, keras_file)

    nose.tools.assert_almost_equal(err, expected_mse, places=4)
