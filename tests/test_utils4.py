from  keras_models_factory.utils4 import hash_array_sum as hash_array
import numpy as np
import pandas as pd
from  keras_models_factory.utils4 import number_of_digits_post_decimal
class TestUtils4(object):

  def test_hash_array_np(self):
    nb_samples = 5

    np.random.seed(0)
    x1 = np.random.randn(nb_samples)
    np.random.seed(0)
    x2 = np.random.randn(nb_samples)
    assert (x1==x2).all()

    h1 = hash_array(x1)
    h2 = hash_array(x2)
    assert h1==h2

    x3 = x2.copy()
    h3 = hash_array(x3)
    assert h3==h2

  def test_hash_array_pd(self):
    nb_samples = 5

    np.random.seed(0)
    S1 = pd.Series(np.random.randn(nb_samples))
    S2 = pd.Series(np.random.randn(nb_samples))
    x1 = pd.concat({'main': S1, 'new': S2}, axis=1)
    x2 = x1.copy()
    assert (x1==x2).all().all()

    h1 = hash_array(x1)
    h2 = hash_array(x2)
    assert h1==h2

  def test_number_of_digits_post_decimal(self):
    assert number_of_digits_post_decimal(0.7576) == 4
    assert number_of_digits_post_decimal(3.14159) == 5
    assert number_of_digits_post_decimal(0.0169) == 18 # should have been 4!
