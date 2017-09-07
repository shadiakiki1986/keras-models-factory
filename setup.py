from distutils.core import setup

setup(name='keras_models_factory',
      version='0.1',
      packages=['keras_models_factory'],
      install_requires=[
        "sklearn",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",

        # until https://github.com/fchollet/keras/pull/7566/files/f2b66a02067cd5a0bc7291231c5fe59f355ff2ad#r134733445
        # use keras 2.0.6 and not 2.0.7
        "Keras==2.0.6",

        "tensorflow",
        "h5py",
        "scikit-image",
        "jupyter",
        "tensorboard"
      ],
      extras_require={
          'dev': [
              'nose'
          ]
      }
      )
