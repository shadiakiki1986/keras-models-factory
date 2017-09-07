# keras-models-factory
Factory generating keras models, with unit tests proving they work

The repository includes tests showing that the networks perform well on certain datasets

[LICENSE](LICENSE)

## Installation

```bash
pip install https://github.com/shadiakiki1986/keras-models-factory
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
pew new -r requirements.txt -r requirements-dev.txt -d TEST_ML
pew in TEST_ML nosetests --logging-level INFO --nocapture
```
