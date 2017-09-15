from  keras_models_factory import autoencoder, datasets #, utils2

from test_base import TestBase, read_params_yml

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class TestAutoencoderBase(TestBase):

  def _data(self, fit_kwargs, data_cb):

      (Xc_train, Yc_train), (Xc_test, Yc_test) = datasets.ds_2(
          num_train=int(0.7*nb_samples),
          num_test =int(0.3*nb_samples),
          classification=False
        )

      fit_kwargs.update({
        'x': Xc_train,
        'y': Xc_train,
        'validation_data': (Xc_test, Xc_test),
      })

      return fit_kwargs

