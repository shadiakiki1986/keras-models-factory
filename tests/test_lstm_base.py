from  keras_models_factory import lstm, datasets, utils3

from test_base import TestBase, read_params_yml

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from sklearn.model_selection import train_test_split

"""
Base class for LSTM tests
"""
class TestLstmBase(TestBase):

  def _data(self, fit_kwargs, data_callback, look_back):

      X_model, Y = data_callback()

      # for lstm, need to stride
      X_calib = utils3._load_data_strides(X_model, look_back)
      Y_calib = Y[(look_back-1):]
    
      # split train/test
      Xc_train, Xc_test, Yc_train, Yc_test = train_test_split(X_calib, Y_calib, train_size=0.8, shuffle=False)
    
      # print(X_calib.shape, Y_calib.shape, Xc_train.shape, Xc_test.shape, Yc_train.shape, Yc_test.shape)
      # (994, 5, 2) (994, 1) (795, 5, 2) (199, 5, 2) (795, 1) (199, 1)
    
      fit_kwargs.update({
        'x': Xc_train,
        'y': Yc_train,
        'validation_data': (Xc_test, Yc_test),
      })

      return fit_kwargs
