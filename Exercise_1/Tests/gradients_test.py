from Network.net import *
import numpy as np


def check_gradients(network, x, y):
    print("... checking gradients")
    p = network.predict(x)
    network._backward_pass(p, y)

    network.layers[0].average_gradients()
    dw = np.copy(network.layers[0].dw)
    dw2 = np.zeros_like(dw)
    w_saved = np.copy(network.layers[0].w)

    for n in range(w_saved.shape[0]):
        for f in range(w_saved.shape[1]):
            inc = 0.0000001
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

            dw2[n, f] = (loss_plus - loss_minus) / (2 * inc)
    network.layers[0].w = w_saved
    assert (np.abs(dw2 - dw).all() < 0.001)
    print("Gradients checked")


if __name__ == "__main__":
    # Load or create network
    network = Net.load_net("../" + Net.network_filename(1))
    # Create random input
    x = np.random.normal(0, 1, (1, network.layers[0].w.shape[0]))
    y = np.zeros(shape=(1, 10), dtype='int32')
    y[0, 0] = 1
    # Check gradients
    check_gradients(network, x, y)


