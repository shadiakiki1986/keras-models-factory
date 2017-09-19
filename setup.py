from distutils.core import setup

# check that this is a 64-bit platform for tensorflow
# https://stackoverflow.com/a/41084963/4126114
import platform
bit_actual, _ = platform.architecture()
bit_required = '64bit'
if bit_actual != bit_required:
  raise Exception(bit_required + " python required (for tensorflow dependency). Found " + bit_actual)

# proceed with setup
setup(name='keras_models_factory',
      version='0.1',
      url='https://github.com/shadiakiki1986/keras-models-factory',
      packages=[
        'keras_models_factory',
      ],
      install_requires=[
        "sklearn",
        "pandas",
        "numpy",
        "scipy",
#        "matplotlib",

        # until https://github.com/fchollet/keras/pull/7566/files/f2b66a02067cd5a0bc7291231c5fe59f355ff2ad#r134733445
        # use keras 2.0.6 and not 2.0.7
        "Keras==2.0.6",

        "tensorflow",
        # optional # "tensorflow-gpu",
        "h5py",
        "scikit-image",
#        "jupyter",
#        "tensorboard"
        "xxhash",
        "requests",
        "requests-cache",
      ],
      extras_require={
          'dev': [
              'nose'
          ]
      }
      )
