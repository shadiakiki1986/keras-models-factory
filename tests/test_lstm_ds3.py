from  keras_models_factory import lstm, datasets #, utils2

from test_lstm_base import TestLstmBase

"""
Test LSTM factory against ds3 dataset (airline passengers)
"""
class TestLstmDs3(TestLstmBase):

  def test_fit_model_2(self):
    self.setUp()

    epochs = 100
    expected_mse = 0.0140
    places=4
    lstm_dim=[4]

    fit_kwargs = {
      'epochs': epochs,
    }
    look_back = 5
    data_cb = lambda: datasets.ds_3(look_back=look_back)

    model_desc = "ds 3, model 2, epochs %s, mse %s, dim %s"%(epochs, expected_mse, lstm_dim)
    fit_kwargs = self._data(fit_kwargs, data_cb, look_back)
    fit_kwargs['verbose']=2
    fit_kwargs['batch_size']=1

    model_callback = lambda: lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim)

    f = lambda *args: self.assert_fit_model(*args)
    f.description = model_desc

    #self.skip_cache = True
    yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
