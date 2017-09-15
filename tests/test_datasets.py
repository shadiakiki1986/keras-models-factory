from keras_models_factory.datasets import machinelearningmastery, random_data
from test_base import TestBase
from nose.tools import assert_almost_equal
import numpy as np

class TestDatasetsRandomData(TestBase):

  def test_ds_1(self):
    nb_samples = 10
    look_back = 3
    X, y = random_data.ds_1(nb_samples, look_back, seed=42)
    assert_almost_equal(X[0,0], 0.6477, places=4)

class TestDatasetsMlm(TestBase):

  def assert_shape(self, X):
    assert len(X.shape)==2
    assert X.shape[0]>5
    assert X.shape[1]==1

  def test_ds_3(self):
    look_back = 4
    X, _ = machinelearningmastery.ds_3()
    self.assert_shape(X)

  def test_ds_4_main(self):
    look_back = 4
    X, _ = machinelearningmastery.ds_4(False)
    self.assert_shape(X)

  def test_ds_4_diff(self):
    look_back = 4
    X, _ = machinelearningmastery.ds_4(True)
    self.assert_shape(X)

  def test_ds_4_noNan(self):
    look_back = 4
    X, _ = machinelearningmastery.ds_4(True)
    assert not np.isnan(X).any().any()

