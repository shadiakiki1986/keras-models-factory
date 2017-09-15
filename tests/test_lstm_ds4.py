import keras_models_factory as kmf

from test_lstm_base import TestLstmBase
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler
import nose

# helper class
def scale3d(scaler, x):
  ori_shape = x.shape
  x = scaler.fit_transform(x.reshape(x.shape[:-1])).reshape(ori_shape)
  return x

"""
Test LSTM factory against ds4 dataset (shampoo sales)
"""
class TestLstmDs4(TestLstmBase):

  def test_persistence_model_forecast(self):
    X, _ = kmf.datasets.machinelearningmastery.ds_4(False)
    train, test = X[0:-12], X[-12:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
      # make prediction
      predictions.append(history[-1])
      # observation
      history.append(test[i])
    # report performance
    rmse = math.sqrt(mean_squared_error(test, predictions))
    #print(rmse)
    # equivalent to 0.486 with the scale -1 .. +1
    nose.tools.assert_almost_equal(rmse, 136.761, places=3)

  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  params = (
    ( 'model 2',      1, 0.0867, 100),
    ( 'model 2', 999999, 0.8220, 100),
    # slow .. aborted
    # (  'model 2', 1, 0.0140, 3000),

    # batch must be 27 for ALL dataset otherwise keras predict fails
    # https://stackoverflow.com/questions/43702481/why-does-keras-lstm-batch-size-used-for-prediction-have-to-be-the-same-as-fittin/44228505#44228505
    (  'model 3', 27, 0.7772, 100),
  )
  def test_fit_model_2(self):
    self.setUp()

    places=4
    lstm_dim=[4]
    look_back = 2
    data_cb = lambda: kmf.datasets.machinelearningmastery.ds_4(True)

    for model_id, batch_size, expected_mse, epochs in self.params:
      fit_kwargs = { 'epochs': epochs, }
      model_desc = "ds 4, %s, epochs %s, mse %s, dim %s, batch %s"%(model_id, epochs, expected_mse, lstm_dim, batch_size)
      fit_kwargs = self._data(fit_kwargs, data_cb, look_back)
      #fit_kwargs['verbose']=2
      fit_kwargs['batch_size']=batch_size
  
      # scale to -1 .. +1
      # Apply to train/test separately as in tutorial
      scaler = MinMaxScaler(feature_range=(-1, 1))
      fit_kwargs['x'] = scale3d(scaler,fit_kwargs['x'])
      fit_kwargs['y'] = scaler.fit_transform(fit_kwargs['y'])
      (Xc_test,Yc_test) = fit_kwargs['validation_data']
      Xc_test = scale3d(scaler,Xc_test)
      Yc_test = scaler.fit_transform(Yc_test)
      fit_kwargs['validation_data'] = (Xc_test,Yc_test)

      # keras cannot do validation with stateful=True unless my code gets more sophisticated
      if model_id=='model 3':
        fit_kwargs['validation_data']=()

      # continue
      batch_input_shape = (batch_size, fit_kwargs['x'].shape[1], fit_kwargs['x'].shape[2])
      model_callback = lambda: kmf.models.lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim) if model_id=='model 2' else kmf.models.lstm.model_3(fit_kwargs['x'].shape[2], lstm_dim, batch_input_shape)
      #model_callback().summary()
  
      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc
  
      if model_id=='model 3': self.skip_cache = True
      yield f, model_callback, fit_kwargs, expected_mse, places, self._model_path



