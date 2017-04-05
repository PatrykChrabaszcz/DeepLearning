from base import *
import lasagne
import matplotlib.pyplot as plt
import theano.tensor as T

if __name__ == "__main__":
    input_var = T.tensor4("input_var")
    network = create_network(input_var=input_var, n_out=40)
    load_net_params(network, 'net_all')
    params = lasagne.layers.get_all_param_values(network)

    # Filters from the first conv layer
    filters_one = params[0]
    print(params[0].shape)

    fig, axes = plt.subplots(nrows=8, ncols=8)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    min = np.inf
    max = -np.inf
    for i in range(64):
        img = filters_one[i]
        min = np.min(min, img.min())
        max = np.max(max, img.max())

    for i in range(64):
        img = filters_one[i]
        img = img.transpose(1, 2, 0)
        img -= min#img.min()
        img /= max#img.max()
        axes.flatten()[i].imshow(img)

    plt.axis('off')
    plt.show()


