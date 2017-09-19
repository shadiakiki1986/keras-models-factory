import keras_models_factory as kmf

from test_lstm_base import TestLstmBase

from keras.callbacks import Callback
class ResetStatesCallback(Callback):
  def __init__(self, max_len:int):
    self.counter = 0
    self.max_len = max_len

  def on_batch_begin(self, batch, logs={}):
    if self.counter % self.max_len == 0:
        self.model.reset_states()
    self.counter += 1
      

"""
Test LSTM factory against random.ds_3 dataset (long-term memory)
"""
class TestLstmRandomDs3(TestLstmBase):

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  params = (
    # stateless lstm, with batch size = 1, does not converge
    # (  'model 2',    1, 0.5558 ),
    # stateless lstm, with batch size > 1, was supposed to converge, no?
    # (  'model 2', 9999, 0.7742 ),
    # stateful  lstm, with batch size = 1, should converge!
    (  'model 3',    1, 0.5322 ),
    # stateful  lstm, with batch size > 1, should also converge
    # (  'model 3',   76, 0.7695 ),
  )
  def test_fit_model_2(self):
    self.setUp()

    epochs = 100
    places=4
    lstm_dim=[4]

    fit_kwargs = {
      'epochs': epochs,
    }
    look_back = 5
    fit_kwargs = self._data(
      fit_kwargs,
      data_callback = lambda: kmf.datasets.random_data.ds_3(20,look_back),
      look_back=look_back
    )
    fit_kwargs['verbose']=2

    for model_id, batch_size, expected_mse in self.params:
      model_desc = "ds_3, %s, batch %s, epochs %s, mse %s, dim %s"%(model_id, batch_size, epochs, expected_mse, lstm_dim)

      fit_kwargs['batch_size']=batch_size

      # keras cannot do validation with stateful=True unless my code gets more sophisticated (by using generators)
      if model_id=='model 3':
        fit_kwargs['validation_data']=()

      # continue
      batch_input_shape = (batch_size, fit_kwargs['x'].shape[1], fit_kwargs['x'].shape[2])
      model_callback = lambda: kmf.models.lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim) if model_id=='model 2' else kmf.models.lstm.model_3(fit_kwargs['x'].shape[2], lstm_dim, batch_input_shape)

      f = lambda *args: self.assert_fit_model(*args)
      f.description = model_desc

      #self.skip_cache = True

      # append the reset-states callback
      # PS 1: if skip_cache=True, the callbacks array is erased in test_base.py#L163
      # PS 2: since I'm setting callbacks here, the tensorboard and checkpoint callbacks wont be used
      fk2 = fit_kwargs.copy()
      # print([k for k in fk2.keys()])
      if model_id=='model 3': fk2['callbacks'] = [ResetStatesCallback(look_back)]
 
      yield f, model_callback, fk2, expected_mse, places, self._model_path
