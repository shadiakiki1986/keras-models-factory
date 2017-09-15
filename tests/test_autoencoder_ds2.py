import  keras_models_factory as kmf

from test_base import TestBase, read_params_yml
from test_autoencoder_base import TestAutoencoderBase

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class TestAutoencoderDs2(TestAutoencoderBase):

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators

  params_1 = read_params_yml(os.path.join(dir_path,'data','params_ae_1.yml'))

  def test_fit_model_1(self):
    self.setUp()
    for nb_samples, epochs, expected_mse, places, ae_dim in self.params_1:
      model_desc = "model_1: nb %s, epochs %s, mse %s, dim %s"%(nb_samples, epochs, expected_mse, ae_dim)

      fit_kwargs = {'epochs': epochs}
      data_cb = lambda: kmf.datasets.random_data.ds_2(
          num_train=int(0.7*nb_samples),
          num_test =int(0.3*nb_samples),
          classification=False
        )

      fit_kwargs = self._data(fit_kwargs, data_cb)
      print(fit_kwargs['x'].shape)
      model_callback = lambda: kmf.models.autoencoder.model_1(fit_kwargs['x'].shape[1], ae_dim)
      #fit_kwargs['verbose']=2

      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc

      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path


  #-------------------------
  params_2 = read_params_yml(os.path.join(dir_path,'data','params_ae_2.yml'))
  def test_fit_model_2(self):
    self.setUp()
    for nb_samples, epochs, expected_mse, places, ae_dim in self.params_2:
      model_desc = "model 2: nb %s, epochs %s, mse %s, dim %s"%(nb_samples, epochs, expected_mse, ae_dim)

      fit_kwargs = {'epochs': epochs}
      data_cb = lambda: kmf.datasets.random_data.ds_2(
          num_train=int(0.7*nb_samples),
          num_test =int(0.3*nb_samples),
          classification=False
        )
      fit_kwargs = self._data(fit_kwargs, data_cb)
      model_callback = lambda: kmf.models.autoencoder.model_2(fit_kwargs['x'].shape[1], ae_dim[0], ae_dim[1], ae_dim[2], ae_dim[3] if len(ae_dim)>=4 else None, True)

      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc

      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
