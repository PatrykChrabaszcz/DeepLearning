from __future__ import print_function

import os
import pickle

import numpy as np
import lasagne
from lasagne.nonlinearities import rectify, sigmoid
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, MaxPool2DLayer

from lasagne.init import HeNormal
from lasagne.layers import Conv2DLayer as ConvLayer

data_path = '../../Celeb_data'


def load_data():
    train_images = np.load(os.path.join(data_path, "train_images_32.npy")) / np.float32(255)
    train_labels = np.uint8(np.load(os.path.join(data_path, "train_labels_32.npy")))
    val_images = np.load(os.path.join(data_path, "val_images_32.npy")) / np.float32(255)
    val_labels = np.uint8(np.load(os.path.join(data_path, "val_labels_32.npy")))
    test_images = np.load(os.path.join(data_path, "test_images_32.npy")) / np.float32(255)
    test_labels = np.uint8(np.load(os.path.join(data_path, "test_labels_32.npy")))

    pixel_mean = np.mean(train_images, axis=0)
    train_images -= pixel_mean
    val_images -= pixel_mean
    test_images -= pixel_mean

    train_mirror = train_images[:, :, :, ::-1]
    train_mirror_labels = train_labels

    train_images = np.concatenate((train_images, train_mirror), axis=0)
    train_labels = np.concatenate((train_labels, train_mirror_labels), axis=0)

    with open(os.path.join(data_path, "attr_names.txt")) as f:
        attr_names = f.readlines()[0].split()

    return dict(
        X_train=train_images,
        Y_train=train_labels,
        X_val=val_images,
        Y_val=val_labels,
        X_test=test_images,
        Y_test=test_labels,
        attr_names=attr_names
    )


def batch_generator(inputs, targets, batch_size, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        # Copied from github.com
        if augment:
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batch_size,2))
            for r in range(batch_size):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def create_network(input_var, n_out):
    network = InputLayer(shape=(256, 3, 32, 32),
                         input_var=input_var)

    network = ConvLayer(network, num_filters=64, filter_size=(7, 7), nonlinearity=rectify,
                        W=HeNormal(gain='relu'), pad='same')
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    network = ConvLayer(network, num_filters=128, filter_size=(3, 3), nonlinearity=rectify,
                        W=HeNormal(gain='relu'), pad='same')
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    network = ConvLayer(network, num_filters=128, filter_size=(3, 3), nonlinearity=rectify,
                        W=HeNormal(gain='relu'), pad='same')
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    network = DenseLayer(network, num_units=512, nonlinearity=rectify)
    network = DropoutLayer(network, p=0.5)
    network = DenseLayer(network, num_units=256, nonlinearity=rectify)
    network = DropoutLayer(network, p=0.5)
    network = DenseLayer(network, num_units=n_out, nonlinearity=sigmoid)
    return network


def save_net_params(network, filename):
    with open(filename, 'wb') as f:
        p = pickle.Pickler(f)
        p.dump(lasagne.layers.get_all_param_values(network))


def load_net_params(network, filename):
    with open(filename, 'rb') as f:
        p = pickle.Unpickler(f)
        lasagne.layers.set_all_param_values(network, p.load())
