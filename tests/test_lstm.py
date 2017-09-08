from  keras_models_factory import lstm, datasets #, utils2

from test_base import TestBase

class TestLstm(TestBase):


  #-------------------------
  params_1 = (
#    # SLOW TEST
#    (int(10e3),  600, 0.0241, [10]),
#    (int(10e3),  600, 0.0147, [10,10]),
#    (int(10e3),  600, 0.0173, [10,10,10]),
#    (int(10e3),  600, 0.0528, [10,10,10,10]),
#    (int(10e3),  600, 0.0093, [30]),
#    (int(10e3),  600, 0.0097, [60]),
#    (int(10e3),  600, 0.0061, [90]),
#    (int(10e3),  600, 0.0146, [30,10]),
#    (int(10e3),  600, 0.0082, [30,30]),
#    (int(10e3),  600, 0.0085, [30,60]),
#    (int(10e3),  600, 0.0079, [60,30]),
#    (int(10e3),  600, 0.0192, [60,60]),
#    (int(10e3),  600, 0.0054, [90,60]),
#    (int(10e3),  600, 0.0086, [90,60,30]),

#    # tests with less epochs
#    (int(10e3),  400, 0.01, [90,60,30]),
#    (int(10e3),  400, 0.01, [30,30]),
#    (int(10e3),  300, 0.0129, [60,30]),

#    # failed tests
#    # stuck since epoch 400 # (int(10e3), 1000, 0.01, [30,20,10]),

#    # tests with less data but more epochs
#    (int( 1e3), 3000, 0.0059, [30]),
#    (int( 1e3), 2100, 0.0112, [60]),
#    (int( 1e3), 4000, 0.0097, [30, 20, 10]),

    # quick tests
    (int( 1e3),  30, 0.7699, [30]),
    (int( 1e3), 100, 0.4471, [30]),
    (int( 1e3),  20, 0.7790, [60]),

    # slower tests
    (int( 1e3), 300, 0.1592, [30]),
    (int( 1e3), 300, 0.0530, [60]),
    (int( 1e3), 300, 0.0847, [60, 30]),
    (int( 1e3), 300, 0.0747, [90, 60, 30]),

  )

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
    for nb_samples, epochs, expected_mse, lstm_dim in self.params_1:
      print("model_1: nb %s, epochs %s"%(nb_samples, epochs))
      print("model_1: mse %s, dim %s"%(expected_mse, lstm_dim))

      fit_kwargs = self._ds_1(epochs, nb_samples)
      model_callback = lambda: lstm.model_1(fit_kwargs['x'].shape[2], lstm_dim)
      # model = utils2.build_lstm_ae(Xc_train.shape[2], lstm_dim[0], look_back, lstm_dim[1:], "adam", 1)

      yield self.assert_fit_model, model_callback, fit_kwargs, expected_mse


  #-------------------------
  params_2 = (
#    # SLOW TEST
#    (int(10e3),  600, 0.0241, [10]),
#    (int(10e3),  600, 0.0147, [10,10]),
#    (int(10e3),  600, 0.0173, [10,10,10]),
#    (int(10e3),  600, 0.0528, [10,10,10,10]),
#    (int(10e3),  600, 0.0093, [30]),
#    (int(10e3),  600, 0.0097, [60]),
#    (int(10e3),  600, 0.0061, [90]),
#    (int(10e3),  600, 0.0146, [30,10]),
#    (int(10e3),  600, 0.0082, [30,30]),
#    (int(10e3),  600, 0.0085, [30,60]),
#    (int(10e3),  600, 0.0079, [60,30]),
#    (int(10e3),  600, 0.0192, [60,60]),
#    (int(10e3),  600, 0.0054, [90,60]),
#    (int(10e3),  600, 0.0086, [90,60,30]),

#    # tests with less epochs
#    (int(10e3),  400, 0.01, [90,60,30]),
#    (int(10e3),  400, 0.01, [30,30]),
#    (int(10e3),  300, 0.0129, [60,30]),

#    # failed tests
#    # stuck since epoch 400 # (int(10e3), 1000, 0.01, [30,20,10]),

#    # tests with less data but more epochs
#    (int( 1e3), 3000, 0.0058, [30]),
#    (int( 1e3), 2100, 0.0110, [60]),
#    (int( 1e3), 4000, 0.01, [30, 20, 10]),

    # quick tests
    (int( 1e3),  30, 0.7862, [30]),
    (int( 1e3), 100, 0.4384, [30]),
    (int( 1e3),  20, 0.7576, [60]),

    # slower tests
#    (int( 1e3), 300, 0.1592, [30]),
#    (int( 1e3), 300, 0.0433, [60]),
#    (int( 1e3), 300, 0.0629, [60, 30]),
#    (int( 1e3), 300, 0.0291, [90, 60, 30]),

  )

  #-------------------------
  def test_fit_model_2(self):
    for nb_samples, epochs, expected_mse, lstm_dim in self.params_2:
      print("model 2: nb %s, epochs %s"%(nb_samples, epochs))
      print("model 2: mse %s, dim %s"%(expected_mse, lstm_dim))

      fit_kwargs = self._ds_1(epochs, nb_samples)
      model_callback = lambda: lstm.model_2(fit_kwargs['x'].shape[2], lstm_dim)

      yield self.assert_fit_model, model_callback, fit_kwargs, expected_mse
