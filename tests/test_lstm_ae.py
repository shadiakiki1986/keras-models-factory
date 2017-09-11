from  keras_models_factory import lstm_ae, datasets

from test_base import TestBase, read_params_yml

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class TestAutoencoder(TestBase):


  #-------------------------
  def _compile(self, model):
    # https://github.com/fchollet/keras/blob/master/tests/integration_tests/test_vector_data_tasks.py#L84
    model.compile(loss="mean_squared_error", optimizer='adam')
    # model.compile(loss="mean_squared_error", optimizer='nadam')
    # model.compile(loss="hinge", optimizer='adagrad')
    return model

  def _data(self, epochs, nb_samples):
      fit_kwargs = {
        'epochs': epochs,
      }

      (Xc_train, Yc_train), (Xc_test, Yc_test) = datasets.ds_1(nb_samples=nb_samples, look_back=5, seed=42)

      fit_kwargs.update({
        'x': Xc_train,
        'y': Xc_train,
        'validation_data': (Xc_test, Xc_test),
      })

      return fit_kwargs

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  params_1 = read_params_yml(os.path.join(dir_path,'data','params_lstm_ae_1.yml'))
  def test_fit_model_1(self):
    self.setUp()
    for nb_samples, epochs, expected_mse, places, mkw in self.params_1:
      model_desc = "model 2: nb %s, epochs %s, mse %s, dim %s"%(nb_samples, epochs, expected_mse, mkw)

      fit_kwargs = self._data(epochs, nb_samples)
      model_callback = lambda: lstm_ae.model_1(fit_kwargs['x'].shape[2], mkw['lstm_dim'], mkw['look_back'], mkw['enc_dim'], fit_kwargs['x'].shape[2])

      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc

      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
