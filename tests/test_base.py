from  keras_models_factory import utils3, utils4, utils

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from hashlib import md5

from keras.models import load_model
from os import path, makedirs
import json
from keras import backend as K
import random as rn
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import nose

# https://stackoverflow.com/a/22721724/4126114
# with local modifications
from collections import OrderedDict
def sortOD(od):
  if isinstance(od,list):
    return [sortOD(x) for x in od]

  if not isinstance(od, dict):
    return od

  res = OrderedDict()
  for k, v in sorted(od.items()):
    res[k] = sortOD(v)
  return res

class TestBase(object): #unittest.TestCase): # https://stackoverflow.com/questions/6689537/nose-test-generators-inside-class#comment46280717_11093309

  #-------------------------
  skip_cache=False
  def setUp(self):
    self._model_path = path.join("/mnt/ec2vol", "test-ml-cache")

    # How can I obtain reproducible results using Keras during development?
    # https://github.com/fchollet/keras/blob/0ffba624c5310fd8b536b516a0c10e23f3a402fa/docs/templates/getting-started/faq.md#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ['PYTHONHASHSEED'] = '0'
    # https://stackoverflow.com/a/34306306/4126114
    np.random.seed(42)
    tf.set_random_seed(1234)

    # the below will not disable parallelism but also GPU
    # https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res#comment79179687_42022950
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) # 2,5
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

  def setupBeforeClass(cls):
    # make sure that GPU is in use
    x = device_lib.list_local_devices()
    assert x[1].name=='/gpu:0'

  def tearDown(self):
    # reset tensorflow session
    # https://github.com/fchollet/keras/blob/master/keras/utils/test_utils.py#L158
    if K.backend() == 'tensorflow':
      K.clear_session()

  #-------------------------
  # get initial epoch
  def _get_initial_epoch(self,tb_log_dir):
    if not path.exists(tb_log_dir):
        #print("tensorboard history not found: %s"%(tb_log_dir))
        return 0

    #print("tensorboard history found: %s"%(tb_log_dir))
    latest = utils3.load_tensorboard_latest_data(tb_log_dir)
    if latest is None:
        #print("tensorboard history is empty")
        return 0

    initial_epoch = latest['step']+1 # 0-based
    #print(
    #    "found history on trained model: epochs: %i, loss: %s, val_loss: %s" %
    #    (initial_epoch, latest['loss'], latest['val_loss'])
    #)

    return initial_epoch

  #-------------------------
  #  epochs = 300
  #  look_back = 5
  def get_callbacks(self, model_file:str, keras_file:str):

    # callbacks
    # CANNOT USE EARLY STOPPING IN TESTS because it borks the reproducibility
    #early_stopping = EarlyStopping(monitor='val_loss',
    #                           patience=100)
    checkpointer = ModelCheckpoint(filepath=keras_file,
                               verbose=0, #2
                               save_best_only=True)
    # https://stackoverflow.com/a/43549608/4126114
    tb_log_dir = path.join(model_file, 'tb')
    tensorboard = TensorBoard(log_dir=tb_log_dir,
                     histogram_freq=10,
                     write_graph=True,
                     write_images=False)

    return tb_log_dir, [checkpointer, tensorboard] # early_stopping, 

  #--------------------
  # callback: lambda function without any parameters and returning a keras model
  # fit_kwargs: args passed to fit. Kind of is a description of training that will be done
  #
  # The callback result model_file is an md5 hash
  # that is unique per model.config and fit_kwargs
  # It is a unique ID for caching
  def _model(self, callback, fit_kwargs, model_path):
    fk2 = fit_kwargs.copy()
    fk2['x'] = utils4.hash_array_sum(fit_kwargs['x'])
    fk2['y'] = utils4.hash_array_sum(fit_kwargs['y'])
    fk2['validation_data'] = [utils4.hash_array_sum(x) for x in fit_kwargs['validation_data']]
    fk2['callbacks'] = [] # ignore model callbacks
    fk2 = sortOD(fk2)

    fk2 = json.dumps(fk2).encode('utf-8')
    fk2 = md5(fk2).hexdigest()

    model = callback()
    mf2 = model.get_config()
    mf2 = sortOD(mf2)
    mf2 = json.dumps(mf2).encode('utf-8')
    mf2 = md5(mf2).hexdigest()

    model_file = path.join(model_path, mf2, fk2)
    #print("model file", model_file)

    # create folders in model_file
    makedirs(model_file, exist_ok=True)

    # proceed
    keras_file = path.join(model_file, 'keras')
    #print("keras file", keras_file)

    self._skip_cache_to_console(keras_file)
    if self.skip_cache or not path.exists(keras_file):
      #print("launch new model")
      return model, model_file, keras_file

    #print("load pre-trained model")
    model = load_model(keras_file)
    # model.summary()
    return model, model_file, keras_file

  def _skip_cache_to_console(self,fn:str):
    if self.skip_cache: print("will skip cache "+fn)

  # expected_mse: expected mean square error. Note that the precision asserted is the same as the precision of this number (check `places` argument in `assert_almost_equal` below)
  def assert_fit_model(self, model_callback, fit_kwargs, expected_mse, places, model_path):

    model, model_file, keras_file = self._model(model_callback, fit_kwargs, model_path)

    model = self._compile(model)
    # model.summary()

    tb_log_dir, callbacks = self.get_callbacks(model_file, keras_file)
    if self.skip_cache:
      self._skip_cache_to_console(': keras callbacks')
      callbacks=[]

    # update
    # http://stackoverflow.com/questions/38987/ddg#26853961
    self._skip_cache_to_console(tb_log_dir)

    initial_epoch = self._get_initial_epoch(tb_log_dir) if not self.skip_cache else 0
    if 'verbose' not in fit_kwargs: fit_kwargs.update({'verbose': 0})
    if 'batch_size' not in fit_kwargs: fit_kwargs.update({'batch_size': int(1e3)})
    if 'callbacks' not in fit_kwargs: fit_kwargs.update({'callbacks': callbacks})
    if 'initial_epoch' not in fit_kwargs: fit_kwargs.update({'initial_epoch': initial_epoch})
    if 'shuffle' not in fit_kwargs: fit_kwargs.update({'shuffle': False})

    history=None
    if fit_kwargs['initial_epoch']!=fit_kwargs['epochs']:
      history = model.fit(**fit_kwargs)
    
    pred_file = path.join(model_file, 'pred.npy')
    self._skip_cache_to_console(pred_file)

    if not self.skip_cache and path.exists(pred_file):
      pred = np.load(pred_file)
    else:
      pred_kwargs = {
        'x': fit_kwargs['x'],
        'verbose': 0,
      }
      if 'batch_size' in fit_kwargs:
        pred_kwargs['batch_size'] = fit_kwargs['batch_size']

      pred = model.predict(**pred_kwargs)
      np.save(pred_file, pred)

    err = utils.mse(fit_kwargs['y'], pred)

    # https://github.com/fchollet/keras/blob/master/tests/integration_tests/test_vector_data_tasks.py#L87
    #assert history.history['val_loss'][-1] < 0.01
    #assert history.history['val_loss'][-1] > 0


    # with 10e3 points
    #      np.linalg.norm of data = 45
    #      and a desired mse <= 0.01
    # The minimum loss required = (45 * 0.01)**2 / 10e3 ~ 2e-5
    #
    # with 1e3 points
    #      np.linalg.norm of data = 14
    #      and a desired mse <= 0.01
    # The minimum loss required = (14 * 0.01)**2 / 1e3 ~ 2e-5 (also)
    nose.tools.assert_almost_equal(err, expected_mse, places=places)

  def _compile(self, model):
    # https://github.com/fchollet/keras/blob/master/tests/integration_tests/test_vector_data_tasks.py#L84
    model.compile(loss="mean_squared_error", optimizer='adam')
    # model.compile(loss="mean_squared_error", optimizer='nadam')
    # model.compile(loss="hinge", optimizer='adagrad')
    return model



import yaml
def read_params_yml(fn):
  with open(fn,'r') as fh:
    x=yaml.load(fh)
    #print(fn,x)
    if x is None: return tuple([])
    x=tuple([tuple(y) for y in x if y is not None])
    return x
