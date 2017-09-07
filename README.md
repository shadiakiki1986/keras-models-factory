# keras-models-factory
Factory generating keras models, with unit tests proving they work

The repository includes tests showing that the networks perform well on certain datasets

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
m1 = lstm.model_1(...)
m2 = lstm.model_2(...)
```

## Testing

```
pip3 install pew
pew new -r requirements-dev.txt KERAS_MODELS_FACTORY
python setup.py install
nosetests --logging-level INFO --nocapture
```
