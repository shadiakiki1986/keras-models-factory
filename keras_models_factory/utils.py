# https://gist.github.com/shadiakiki1986/2c293e364563492c65bffdb6122b4e92
from sklearn.preprocessing import MinMaxScaler #  normalize,
min_max_scaler = MinMaxScaler()
# def myNorm3(X): return normalize(X, norm='l2', axis=0)
def myNorm3(X): return min_max_scaler.fit_transform(X)

##########################################
import numpy as np
from matplotlib import pyplot as plt
def myPlot(X, space:int=5):
    X_plt = X+space*np.arange(X.shape[1])
    N_PLOT=200
    plt.plot(X_plt[0:N_PLOT,:])
    plt.show()


from sklearn.model_selection import train_test_split
def ae_fit_encode_plot_mse(X_in, autoencoder, encoder, N_epochs, verbose=1, callbacks:list=[]):
  # split
  X_train, X_test = train_test_split(X_in, train_size=0.8, random_state=8888)

  # train autoencoder
  autoencoder.fit(
    X_train,
    X_train,
    epochs=N_epochs,
    batch_size=256,
    shuffle=True,
    validation_data=(
      X_test,
      X_test,
    ),
    verbose = verbose,
    callbacks=callbacks
  )

  # if not easy to visualize
  if X_in.shape[1]<50:
    # print("encoder predict")
    X_enc = encoder.predict(X_in)
    # print("encoded",X_enc)
    # # X_enc_dec = decoder.predict(X_enc)
    # # print("enc-dec",X_enc_dec)
    # X_rec = autoencoder.predict(X_pca)
    # print("recoded",X_rec)

    # plot
    # from matplotlib import pyplot as plt
    myPlot(X_enc)

  X_rec = autoencoder.predict(X_in)

  #result = mse(X_in, X_rec)
  #print("AE mse = ", result)
  #return result
  return X_rec

#####################
# functions for t1e_pca_ae_nonlinear-2
# copied from https://stats.stackexchange.com/questions/190148/autoencoder-pca-tensorflow?rq=1
def mse(x, x_est):
    numerator = np.linalg.norm(x - x_est)
    denominator = np.linalg.norm(x)
    #print('num/deonm', numerator, denominator, numerator/denominator)
    return numerator/denominator

from sklearn.linear_model import LinearRegression
def pca_err(X, x_pca):
    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=2).fit(X)
    #x_pca = pca.transform(X)
    lr = LinearRegression().fit(x_pca, X)
    x_est = lr.predict(x_pca)
    result = mse(X, x_est)
    print('err pca = ', result)
    return result
