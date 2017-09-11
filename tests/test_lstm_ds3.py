from  keras_models_factory import lstm, datasets #, utils2

from test_base import TestBase, read_params_yml

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

"""
Test LSTM factory against ds3 dataset (airline passengers)
"""
class TestLstmDs3(TestBase):

  def _compile(self, model):
    # https://github.com/fchollet/keras/blob/master/tests/integration_tests/test_vector_data_tasks.py#L84
    model.compile(loss="mean_squared_error", optimizer='adam')
    # model.compile(loss="mean_squared_error", optimizer='nadam')
    # model.compile(loss="hinge", optimizer='adagrad')
    return model

  def _data(self, epochs):
      fit_kwargs = {
        'epochs': epochs,
      }

      (Xc_train, Yc_train), (Xc_test, Yc_test) = datasets.ds_3(look_back=5)

      fit_kwargs.update({
        'x': Xc_train,
        'y': Yc_train,
        'validation_data': (Xc_test, Yc_test),
      })

      return fit_kwargs

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  def test_fit_model_2(self):
    self.setUp()

    epochs = 100
    expected_mse = 0.0416
    places=4
    lstm_dim=[4]

    model_desc = "ds 3, model 2, epochs %s, mse %s, dim %s"%(epochs, expected_mse, lstm_dim)
    fit_kwargs = self._data(epochs)
    fit_kwargs['verbose']=2
    fit_kwargs['batch_size']=2

    model_callback = lambda: lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim)

    f = lambda *args: self.assert_fit_model(*args)
    f.description = model_desc

    #self.skip_cache = True
    yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
