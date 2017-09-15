from  keras_models_factory import autoencoder, datasets #, utils2

from test_base import TestBase, read_params_yml

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from sklearn.model_selection import train_test_split

class TestAutoencoderBase(TestBase):

  def _data(self, fit_kwargs, data_cb):

      X_calib, Y_calib = data_cb()

      # split train/test
      Xc_train, Xc_test, Yc_train, Yc_test = train_test_split(X_calib, Y_calib, train_size=0.8, shuffle=False)

      fit_kwargs.update({
        'x': Xc_train,
        'y': Xc_train,
        'validation_data': (Xc_test, Xc_test),
      })

      return fit_kwargs

