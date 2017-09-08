from  keras_models_factory import lstm, utils, datasets #, utils2

import nose

from test_base import TestBase

class TestLstm(TestBase):

  #  epochs = 300
  def _fit(self, Xc_train, Xc_test, Yc_train, Yc_test, model, epochs:int, model_file:str, keras_file:str):
    if epochs<=0: raise Exception("epochs <= 0")


    tb_log_dir, callbacks = self.get_callbacks(model_file, keras_file)
    history = model.fit(
        x=Xc_train,
        y=Yc_train,
        epochs = epochs,
        verbose = 0,#2,
        batch_size = 1000, # 100
        validation_data = (Xc_test, Yc_test),
        callbacks = callbacks,
        initial_epoch = self._get_initial_epoch(tb_log_dir),
        shuffle=False
    )
    
    pred = model.predict(x=Xc_train, verbose = 0)

    err = utils.mse(Yc_train, pred)

    return (history, err)

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

  #-------------------------
  # http://nose.readthedocs.io/en/latest/writing_tests.html#test-generators
  def test_fit_model_1(self):
    for nb_samples, epochs, expected_mse, lstm_dim in self.params_1:
      yield self.check_fit_model_1, nb_samples, epochs, expected_mse, lstm_dim

  def check_fit_model_1(self, nb_samples, epochs, expected_mse, lstm_dim):
    train_desc = "model_1: nb %s, epochs %s"%(nb_samples, epochs)
    print(train_desc)
    print("model_1: mse %s, dim %s"%(expected_mse, lstm_dim))

    (Xc_train, Yc_train), (Xc_test, Yc_test) = datasets.ds_1(nb_samples=nb_samples, look_back=5)

    model, model_file, keras_file = self._model(lambda: lstm.model_1(Xc_train.shape[2], lstm_dim), train_desc)
    # model = utils2.build_lstm_ae(Xc_train.shape[2], lstm_dim[0], look_back, lstm_dim[1:], "adam", 1)

    model = self._compile(model)
    # model.summary()

    (history, err) = self._fit(Xc_train, Xc_test, Yc_train, Yc_test, model, epochs, model_file, keras_file)

    # with 10e3 points
    #      np.linalg.norm of data = 45
    #      and a desired mse <= 0.01
    # The minimum loss required = (45 * 0.01)**2 / 10e3 ~ 2e-5
    #
    # with 1e3 points
    #      np.linalg.norm of data = 14
    #      and a desired mse <= 0.01
    # The minimum loss required = (14 * 0.01)**2 / 1e3 ~ 2e-5 (also)
    nose.tools.assert_almost_equal(err, expected_mse, places=4)

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
    (int( 1e3),  30, 0.7699, [30]),
    (int( 1e3), 100, 0.4471, [30]),
    (int( 1e3),  20, 0.7790, [60]),

    # slower tests
    (int( 1e3), 300, 0.1592, [30]),
    (int( 1e3), 300, 0.0530, [60]),
    (int( 1e3), 300, 0.0417, [60, 30]),
    (int( 1e3), 300, 0.0296, [90, 60, 30]),

  )

  #-------------------------
  def test_fit_model_2(self):
    for (nb_samples, epochs, expected_mse, lstm_dim) in self.params_2:
      yield self.check_fit_model_2, nb_samples, epochs, expected_mse, lstm_dim
      #self.check_fit_model_2( nb_samples, epochs, expected_mse, lstm_dim )

  def check_fit_model_2(self, nb_samples, epochs, expected_mse, lstm_dim):
    print("model 2: mse %s, dim %s"%(expected_mse, lstm_dim))
    train_desc = "nb %s, epochs %s"%(nb_samples, epochs)
    print(train_desc)
    (Xc_train, Yc_train), (Xc_test, Yc_test) = datasets.ds_1(nb_samples=nb_samples, look_back=5)

    model, model_file, keras_file = self._model(lambda: lstm.model_2(Xc_train.shape[2], lstm_dim), train_desc)
  
    model = self._compile(model)
    # model.summary()

    (history, err) = self._fit(Xc_train, Xc_test, Yc_train, Yc_test, model, epochs, model_file, keras_file)

    # https://github.com/fchollet/keras/blob/master/tests/integration_tests/test_vector_data_tasks.py#L87
    #assert history.history['val_loss'][-1] < 0.01
    #assert history.history['val_loss'][-1] > 0
    nose.tools.assert_almost_equal(err, expected_mse, places=4)
