from  keras_models_factory import lstm, datasets #, utils2

from test_base import TestBase, read_params_yml

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class TestLstm(TestBase):


  #-------------------------
  params_1 = read_params_yml(os.path.join(dir_path,'data','params_lstm_1_main.yml'))

  def _compile(self, model):
    # https://github.com/fchollet/keras/blob/master/tests/integration_tests/test_vector_data_tasks.py#L84
    model.compile(loss="mean_squared_error", optimizer='adam')
    # model.compile(loss="mean_squared_error", optimizer='nadam')
    # model.compile(loss="hinge", optimizer='adagrad')
    return model

  def _ds_1(self, epochs, nb_samples):
      fit_kwargs = {
        'epochs': epochs,
      }

      (Xc_train, Yc_train), (Xc_test, Yc_test) = datasets.ds_1(nb_samples=nb_samples, look_back=5, seed=42)

      fit_kwargs.update({
        'x': Xc_train,
        'y': Yc_train,
        'validation_data': (Xc_test, Yc_test),
      })

      return fit_kwargs

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  def test_fit_model_1(self):
    self.setUp()
    for nb_samples, epochs, expected_mse, places, lstm_dim in self.params_1:
      model_desc = "model_1: nb %s, epochs %s, mse %s, dim %s"%(nb_samples, epochs, expected_mse, lstm_dim)

      fit_kwargs = self._ds_1(epochs, nb_samples)
      model_callback = lambda: lstm.model_1(fit_kwargs['x'].shape[2], lstm_dim)
      # model = utils2.build_lstm_ae(Xc_train.shape[2], lstm_dim[0], look_back, lstm_dim[1:], "adam", 1)

      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc

      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path


  #-------------------------
  params_2 = read_params_yml(os.path.join(dir_path,'data','params_lstm_2_main.yml'))

  #-------------------------
  def test_fit_model_2(self):
    self.setUp()
    for nb_samples, epochs, expected_mse, places, lstm_dim in self.params_2:
      model_desc = "model 2: nb %s, epochs %s, mse %s, dim %s"%(nb_samples, epochs, expected_mse, lstm_dim)

      fit_kwargs = self._ds_1(epochs, nb_samples)
      model_callback = lambda: lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim)

      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc

      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
