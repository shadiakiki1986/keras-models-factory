# https://stackoverflow.com/a/31829201/4126114
import xxhash
hasher = xxhash.xxh64()
def hash_array_xxhash(x):
  print('hash array')
  hasher.update(x)
  x = hasher.intdigest()
  hasher.reset()
  return x

import pandas as pd
from math import floor
def hash_array_sum(x):
  if isinstance(x, pd.DataFrame):
    x = x.values
  return floor(1e8*x.sum())

# Find the number of digits after the decimal point
# https://stackoverflow.com/a/35586744/4126114
def number_of_digits_post_decimal(x):  
  count = 0  
  residue = x -int(x)  
  threshold = 1e8
  if residue != 0:  
    multiplier = 1  
#    while not (x*multiplier).is_integer():  
    while not (floor(threshold*x*multiplier)/threshold).is_integer():
      count += 1
      multiplier = 10 * multiplier
#      print(multiplier, count, x*multiplier)
    return count

