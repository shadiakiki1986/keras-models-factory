# could make wrapper from https://gist.github.com/ktrnka/81c8a7b79cb05c577aab
# and make pipeline
# copied from simple example at https://blog.keras.io/building-autoencoders-in-keras.html
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras.layers.advanced_activations import LeakyReLU #, PReLU

# single-layer autoencoder
def model_1(input_shape:int, encoding_dim_ae:int=2):
    input_img = Input(shape=(input_shape,))
    encoded = input_img
    # encoded = Dense( encoding_dim_ae, activation='relu' )(encoded)

    # hidden layer
    encoded = Dense( encoding_dim_ae, activation='linear' )(encoded)

    # use leaky relu
    # https://github.com/fchollet/keras/issues/117
    encoded = LeakyReLU(alpha=.3)(encoded)   # add an advanced activation

    # GET DEEP
    # encoding_dim2 = 50
    # encoding_dim3 = 10
    # encoded = Dense(encoding_dim2, activation='relu')(encoded)
    # encoded = Dense(encoding_dim3, activation='relu')(encoded)
    # decoded = Dense(encoding_dim2, activation='relu')(encoded)

    decoded = Dense(input_shape, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    #encoder = Model(input_img, encoded)

    # encoded_input = Input(shape=(encoding_dim_ae,))
    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    # other: optimizer='adadelta', loss='binary_crossentropy'
    #autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    #encoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    # decoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    
    #return (autoencoder, encoder)
    return autoencoder

# deep autoencoder
def model_2(input_shape:int, enc_dim1:int, enc_dim2:int=None, enc_dim3:int=None, enc_dim4:int=None, symmetric:bool=True):
    if enc_dim2 is None and (enc_dim3 is not None or enc_dim4 is not None):
      raise Exception("dim2 is None but dim3 or dim4 is not None")
    if enc_dim3 is None and enc_dim4 is not None:
      raise Exception("dim3 is None but dim4 is not None")

    input_img = Input(shape=(input_shape,))
    encoded = input_img
    # encoded = Dense( encoding_dim_ae, activation='relu' )(encoded)

    # hidden layer
    encoded = Dense( enc_dim1, activation='linear' )(encoded)

    # use leaky relu
    # https://github.com/fchollet/keras/issues/117
    encoded = LeakyReLU(alpha=.3)(encoded)   # add an advanced activation

    # GET DEEP
    decoded = encoded
    if enc_dim2 is not None:
      encoded = Dense(enc_dim2, activation='linear')(encoded)
      encoded = LeakyReLU(alpha=.3)(encoded)
      decoded = encoded
      if enc_dim3 is not None:
        encoded = Dense(enc_dim3, activation='linear')(encoded)
        encoded = LeakyReLU(alpha=.3)(encoded)
        decoded = encoded
        if enc_dim4 is not None:
          encoded = Dense(enc_dim4, activation='linear')(encoded)
          encoded = LeakyReLU(alpha=.3)(encoded)
          decoded = encoded
          # ...
          if symmetric: decoded = Dense(enc_dim3, activation='relu')(decoded)
        if symmetric: decoded = Dense(enc_dim2, activation='relu')(decoded)
      if symmetric: decoded = Dense(enc_dim1, activation='relu')(decoded)
    decoded = Dense(input_shape, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    #encoder = Model(input_img, encoded)

    # encoded_input = Input(shape=(encoding_dim_ae,))
    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    # other: optimizer='adadelta', loss='binary_crossentropy'
    #autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    #encoder.compile(optimizer=optimizer, loss='mean_squared_error')
    # decoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    
#    return (autoencoder, encoder)
    return autoencoder
