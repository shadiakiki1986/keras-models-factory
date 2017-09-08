from keras_models_factory.datasets import ds_1
from test_base import TestBase
from nose.tools import assert_almost_equal
class TestUtils4(TestBase):

  def test_ds_1(self):
    nb_samples = 10
    look_back = 3
    (xtr,ytr), (xte,yte) = ds_1(nb_samples, look_back, seed=42)
    assert_almost_equal(xtr[0,0,0], 0.6477, places=4)
