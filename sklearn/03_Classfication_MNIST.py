from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)

fetch_mnist()
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

def peek_mnist(num):
    some_digit = X[num]
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image,cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()
    print (y[num])

# peek_mnist(36000)

# shuffle the train data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
