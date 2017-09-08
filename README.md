# keras-models-factory
Factory generating keras models, with unit tests proving they work.

The repository includes integration tests showing that the networks perform well on certain datasets.

The tests include caching so that if a test is interrupted and re-launched,
it resumes from where it had reached.

The idea is also available in the [keras integration tests](https://github.com/fchollet/keras/blob/master/tests/integration_tests/test_image_data_tasks.py)

[LICENSE](LICENSE)

## Installation

```bash
pip3 install pew
pew new TEST_KERAS_MODELS_FACTORY
pip install git+https://github.com/shadiakiki1986/keras-models-factory.git
```

## Usage

To test a keras model (copy one of the integration tests in [tests](tests))

To use a model from the factory:

```python
from keras_models_factory import lstm

# stacked lstm
m1 = lstm.model_1(in_neurons=10, lstm_dim=[8,6], look_back=5)
m1.summary()
# ...

# single lstm followed by dense layers
m2 = lstm.model_2(in_neurons=10, lstm_dim=[8,6], look_back=5)
m2.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_1 (LSTM)                (None, 8)                 608
# _________________________________________________________________
# dense_1 (Dense)              (None, 6)                 54
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 7
# =================================================================
# Total params: 669
# Trainable params: 669
# Non-trainable params: 0
# _________________________________________________________________

```


## Testing

```
sudo apt-get install python3-tk

pip3 install pew
pew new KERAS_MODELS_FACTORY

# http://stackoverflow.com/questions/28509965/ddg#28842733
pip install .[dev]

# Not sure if this is necessary
python setup.py install

# run tests
nosetests --logging-level INFO --nocapture

# or run individual tests
# http://pythontesting.net/framework/specify-test-unittest-nosetests-pytest/
# https://nose.readthedocs.io/en/latest/plugins/logcapture.html
nosetests --logging-level INFO --nocapture -v tests/test_lstm.py:TestLstm.test_fit_model_1
nosetests --logging-level INFO --nocapture -v tests/test_lstm.py:TestLstm.test_fit_model_2

```
