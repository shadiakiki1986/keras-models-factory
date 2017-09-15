import keras_models_factory as kmf

from test_lstm_base import TestLstmBase

"""
Test LSTM factory against ds3 dataset (airline passengers)
"""
class TestLstmDs3(TestLstmBase):

  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  params = (
    (  1, 0.0140, 100),

    # For some reason, increasing the batch size reduces the MSE when using the same number of epochs
    # and requires more epochs to achieve the same MSE
    # https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
    (100, 0.3521, 100),
    (100, 0.1701, 300),
    (100, 0.1245, 600),

    (200, 0.3361, 100),

  )
  def test_fit_model_2(self):
    self.setUp()

    places=4
    lstm_dim=[4]
    look_back = 5
    data_cb = lambda: kmf.datasets.machinelearningmastery.ds_3()

    for batch_size, expected_mse, epochs in self.params:
      fit_kwargs = { 'epochs': epochs, }
      model_desc = "ds 3, model 2, epochs %s, mse %s, dim %s, batch %s"%(epochs, expected_mse, lstm_dim, batch_size)
      fit_kwargs = self._data(fit_kwargs, data_cb, look_back)
      #fit_kwargs['verbose']=2
      fit_kwargs['batch_size']=batch_size
  
      model_callback = lambda: kmf.models.lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim)
      #model_callback().summary()
  
      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc
  
      #self.skip_cache = True
      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path
