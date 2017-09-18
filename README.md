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

For GPU instances (e.g. AWS EC2 p2.xlarge): run [init-gpu.sh](https://gist.github.com/shadiakiki1986/0c9ea999113691fb9a7ae64e3541fe29)

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
pip install -e .[dev]

# run tests
nosetests --logging-level INFO --nocapture

# or run individual tests
# http://pythontesting.net/framework/specify-test-unittest-nosetests-pytest/
# https://nose.readthedocs.io/en/latest/plugins/logcapture.html
nosetests --logging-level INFO --nocapture -v tests/test_lstm_ds1.py:TestLstmDs1.test_fit_model_1
nosetests --logging-level INFO --nocapture -v tests/test_lstm_ds1.py:TestLstmDs1.test_fit_model_2

```

Note that ATM I moved some LSTM slow tests into
- `tests/data/params_lstm_1_slow.yml`
- and `tests/data/params_lstm_2_slow.yml`

To run tests in parallel

```bash
sudo apt-get install parallel
ls tests/test*py -1|pew in KERAS_MODELS_FACTORY parallel nosetests --logging-level INFO --nocapture -v {}
```

Note that the `nosetests` parameter `--processes` doesnt work and yields a `CUDA_ERROR_NOT_INITIALIZED`

## Dev notes
In order for `import keras_models_factor as kmf; help(kmf.datasets)` to work,
I use the `keras_models_factory/__init__.py` to have the imports.
Ref: https://stackoverflow.com/a/46242108/4126114
