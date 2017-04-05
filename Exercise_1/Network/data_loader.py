import os
import gzip
import pickle
import numpy as np


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('... done loading data')
    return rval


def stochastic_data(x, y, batch_size):
    idxs = np.random.permutation(x.shape[0])[:batch_size]
    x = x[idxs]
    y = y[idxs]

    x = np.reshape(x, (x.shape[0], -1))
    y = one_hot(y)
    return x, y


def full_data_generator(x, y, batch_size):
    x = np.reshape(x, (x.shape[0], -1))
    num = x.shape[0]
    iter = int(num/batch_size)
    for i in range(iter):
        x_ = x[i*batch_size:(i+1)*batch_size]
        y_ = y[i*batch_size:(i+1)*batch_size]
        y_ = one_hot(y_)
        yield x_, y_
    if num % batch_size != 0:
        x_ = x[iter*batch_size:num]
        y_ = y[iter*batch_size:num]
        y_ = one_hot(y_)
        yield x_, y_


def full_stochastic_generator(x, y, batch_size):
    # Shuffle
    idxs = np.random.permutation(x.shape[0])
    x = x[idxs]
    y = y[idxs]

    return full_data_generator(x, y, batch_size)


def one_hot(y):
    size = 10
    array = np.zeros(shape=(y.shape[0], size))

    for c in range(size):
        array[y == c, c] = 1

    return array