# keras-models-factory
Factory generating keras models, with unit tests proving they work

The repository includes tests showing that the networks perform well on certain datasets

The tests include caching so that if a test is interrupted and re-launched,
it resumes from where it had reached

[LICENSE](LICENSE)

## Installation

```bash
pip3 install pew
pew new TEST_KERAS_MODELS_FACTORY
pip install git+https://github.com/shadiakiki1986/keras-models-factory.git
```

## Usage

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
pip3 install pew
pew new KERAS_MODELS_FACTORY
python setup.py install

# or
# pip install .[dev]
#  http://stackoverflow.com/questions/28509965/ddg#28842733

nosetests --logging-level INFO --nocapture
```
