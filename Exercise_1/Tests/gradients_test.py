from Network.net import *
from utils import create_network
import numpy as np


def check_gradients(network, x, y):
    print("Checking gradients ...")
    p = network.predict(x)
    network.backward_pass(p, y)

    w_grad_analytical = network.layers[0].grad_w_acc.mean_gradient()
    w_grad_numerical = np.zeros_like(w_grad_analytical)
    w_saved = np.copy(network.layers[0].w)

    for n in range(w_saved.shape[0]):
        for f in range(w_saved.shape[1]):
            inc = 0.01
            # Compute change in loss for positive increment
            network.layers[0].w[n, f] += inc
            p_plus = network.predict(x)
            loss_plus = network.loss(p_plus, y)
            # Compute change in loss for negative increment
            network.layers[0].w[n, f] -= inc * 2
            p_minus = network.predict(x)
            loss_minus = network.loss(p_minus, y)
            # Reverse changes
            network.layers[0].w[n, f] += inc

            w_grad_numerical[n, f] = (loss_plus - loss_minus) / (2 * inc)
    network.layers[0].w = w_saved
    assert (np.abs(w_grad_analytical - w_grad_numerical).all() < 0.001)
    print("Gradients checked")


if __name__ == "__main__":
    # Load or create network
    network = create_network(hidden_size=(20, 20))
    # Create random input
    x = np.random.normal(0, 1, (1, network.layers[0].w.shape[0]))
    y = np.zeros(shape=(1, 10), dtype='int32')
    y[0, 0] = 1
    # Check gradients
    check_gradients(network, x, y)
