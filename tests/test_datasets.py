import keras_models_factory as kmf
from test_base import TestBase
from nose.tools import assert_almost_equal
import numpy as np

class TestDatasetsRandomData(TestBase):

  def test_ds_1(self):
    nb_samples = 10
    look_back = 3
    lags=[1,2]

    if look_back < max(lags):
      raise Exception("Not enough look back provided")

    X, y = kmf.datasets.random_data.ds_1(nb_samples, seed=42, lags=lags)
    assert_almost_equal(X[0,0], 0.6477, places=4)

  def test_ds_3(self):
    X, Y = kmf.datasets.random_data.ds_3(10,5)

    assert len(X.shape)==2
    assert X.shape[0]==50
    assert X.shape[1]==1

    assert len(Y.shape)==2
    assert Y.shape[0]==50
    assert Y.shape[1]==1

    assert X.min()>=-1
    assert X.max()<=+1
    assert Y.min()>=-1
    assert Y.max()<=+1


class TestDatasetsMlm(TestBase):

  def assert_shape(self, X):
    assert len(X.shape)==2
    assert X.shape[0]>5
    assert X.shape[1]==1


  def test_ds_3(self):
    X, _ = kmf.datasets.machinelearningmastery.ds_3()
    self.assert_shape(X)

  def test_ds_4_main(self):
    X, _ = kmf.datasets.machinelearningmastery.ds_4(False)
    self.assert_shape(X)
    assert X.shape[0]==36

  def test_ds_4_diff(self):
    X, _ = kmf.datasets.machinelearningmastery.ds_4(True)
    self.assert_shape(X)

  def test_ds_4_noNan(self):
    X, _ = kmf.datasets.machinelearningmastery.ds_4(True)
    assert not np.isnan(X).any().any()

