from  keras_models_factory import lstm_ae, datasets

from test_base import read_params_yml
from test_lstm_base import TestLstmBase

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class TestLstmAe(TestLstmBase):

  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  params_1 = read_params_yml(os.path.join(dir_path,'data','params_lstm_ae_1.yml'))
  def test_fit_model_1(self):
    self.setUp()
    for nb_samples, epochs, expected_mse, places, mkw in self.params_1:
      model_desc = "model 2: nb %s, epochs %s, mse %s, dim %s"%(nb_samples, epochs, expected_mse, mkw)

      fit_kwargs = {
        'epochs': epochs,
      }
      look_back = 5
      data_cb = lambda: datasets.randn.ds_1(nb_samples=nb_samples, look_back=look_back, seed=42)

      fit_kwargs = self._data(fit_kwargs, data_cb, look_back)

      # since AE
      fit_kwargs['y'] = fit_kwargs['x']
      Xc_test, _ = fit_kwargs['validation_data']
      fit_kwargs['validation_data'] = (Xc_test,Xc_test)

      model_callback = lambda: lstm_ae.model_1(fit_kwargs['x'].shape[2], mkw['lstm_dim'], mkw['look_back'], mkw['enc_dim'], fit_kwargs['x'].shape[2])

      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc

      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
