from keras_models_factory import datasets
from test_base import TestBase
from nose.tools import assert_almost_equal
class TestUtils4(TestBase):

  def test_ds_1(self):
    nb_samples = 10
    look_back = 3
    (xtr,ytr), (xte,yte) = datasets.ds_1(nb_samples, look_back, seed=42)
    assert_almost_equal(xtr[0,0,0], 0.6477, places=4)

  def test_ds_3(self):
    look_back = 4
    (xtr,ytr),(xte,yte) = datasets.ds_3(look_back)

    assert xtr.shape[0]>5
    assert xtr.shape[1]==look_back
    assert xtr.shape[2]==1

    assert len(ytr.shape)==2
    assert ytr.shape[0]>5
    assert ytr.shape[1]==1
