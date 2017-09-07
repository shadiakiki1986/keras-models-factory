# Deps
# sudo apt-get install python3-tk
# pip install nose
#
# Run all
# nosetests --logging-level INFO test_lstm.py
#
# Run a single test class
# http://pythontesting.net/framework/specify-test-unittest-nosetests-pytest/
# https://nose.readthedocs.io/en/latest/plugins/logcapture.html
# nosetests --logging-level INFO --nocapture -v test_lstm.py:TestLstm.test_fit_model_1
# nosetests --logging-level INFO --nocapture -v test_lstm.py:TestLstm.test_fit_model_2


from  keras_models_factory import utils3

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from hashlib import md5

from keras.models import load_model
from os import path, makedirs
import json

# https://stackoverflow.com/a/22721724/4126114
from collections import OrderedDict
def sortOD(od):
  res = OrderedDict()
  for k, v in sorted(od.items()):
    if isinstance(v, dict):
      res[k] = sortOD(v)
    else:
      res[k] = v
  return res

class TestBase(object): #unittest.TestCase): # https://stackoverflow.com/questions/6689537/nose-test-generators-inside-class#comment46280717_11093309

  #-------------------------
  # save model in file: filename is md5 checksum of models file
  # This way, any edits in the file result in a new filename and hence re-calculating the model
  def setUp(self):
    self._model_path = path.join("/", "tmp", "test-ml-cache")

  #-------------------------
  # get initial epoch
  def _get_initial_epoch(self,tb_log_dir):
    if not path.exists(tb_log_dir):
        print("tensorboard history not found: %s"%(tb_log_dir))
        return 0

    print("tensorboard history found: %s"%(tb_log_dir))
    latest = utils3.load_tensorboard_latest_data(tb_log_dir)
    if latest is None:
        print("tensorboard history is empty")
        return 0

    initial_epoch = latest['step']+1 # 0-based
    print(
        "found history on trained model: epochs: %i, loss: %s, val_loss: %s" %
        (initial_epoch, latest['loss'], latest['val_loss'])
    )

    return initial_epoch

  #-------------------------
  #  epochs = 300
  #  look_back = 5
  def get_callbacks(self, model_file:str, keras_file:str):

    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                               patience=100)
    checkpointer = ModelCheckpoint(filepath=keras_file,
                               verbose=0, #2
                               save_best_only=True)
    # https://stackoverflow.com/a/43549608/4126114
    tb_log_dir = path.join(model_file, 'tb')
    tensorboard = TensorBoard(log_dir=tb_log_dir,
                     histogram_freq=10,
                     write_graph=True,
                     write_images=False)

    return tb_log_dir, [early_stopping, checkpointer, tensorboard]

  #--------------------
  # callback: lambda function without any parameters and returning a keras model
  def _model(self,callback):
    model = callback()
    model_file = [sortOD(x) for x in model.get_config()]
    model_file = json.dumps(model_file).encode('utf-8')
    model_file = md5(model_file).hexdigest()
    model_file = path.join(self._model_path, model_file)
    print("model file", model_file)

    # create folders in model_file
    makedirs(model_file, exist_ok=True)

    # proceed
    keras_file = path.join(model_file, 'keras')
    print("keras file", keras_file)
    if not path.exists(keras_file):
      print("launch new model")
      return model, model_file, keras_file

    print("load pre-trained model")
    model = load_model(keras_file)
    # model.summary()
    return model, model_file, keras_file
