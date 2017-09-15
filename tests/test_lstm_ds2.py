from  keras_models_factory.models import lstm
from  keras_models_factory.datasets import random_data

from test_lstm_base import TestLstmBase

"""
Test LSTM factory against ds_2 dataset (keras random matrix)
"""
class TestLstmDs2(TestLstmBase):

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  def test_fit_model_2(self):
    self.setUp()

    epochs = 100
    expected_mse = 34.4729 # HORRIBLE
    places=4
    lstm_dim=[4]

    model_desc = "ds_2, model 2, epochs %s, mse %s, dim %s"%(epochs, expected_mse, lstm_dim)
    fit_kwargs = {
      'epochs': epochs,
    }
    fit_kwargs = self._data(fit_kwargs, lambda: random_data.ds_2(), 5)
    #fit_kwargs['verbose']=2
    fit_kwargs['batch_size']=1

    model_callback = lambda: lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim)

    f = lambda *args: self.assert_fit_model(*args)
    f.description = model_desc

    #self.skip_cache = True
    yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
