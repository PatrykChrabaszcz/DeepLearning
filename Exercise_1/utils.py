from Network.net import Net, Objective, WeightInit, DenseLayer, Activation, Metric
from Network.data_loader import full_data_generator
import numpy as np
import time


class Stats:
    def __init__(self, name='Stats', verbose=True):
        self.name = name
        self.verbose = verbose

        self.start_time = None
        self.duration = 0
        self.children = []

    def create_child_stats(self, name):
        child_stats = Stats(name, verbose=False)
        self.children.append(child_stats)

        return child_stats

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration += time.time() - self.start_time

        if self.verbose:
            print('%s %gs' % (self.name, self.duration))
            for child in self.children:
                print('\t%s took %g%%' % (child.name, child.duration/self.duration*100))


# hidden_size list with number of neurons in each layer
# doesn't include output and input layers
def create_network(hidden_size=()):

    network = Net(objective=Objective.CrossEntropyWithLogits)

    # init_func = WeightInit.gaussian_init
    init_func = WeightInit.xavier_init_gauss

    input_size_info = 784
    for size in hidden_size:
        layer = DenseLayer(activation_func=Activation.Relu, init_func=init_func,
                           input_size_info=input_size_info, neurons_num=size)
        network.add_layer(layer)
        input_size_info = layer

    layer = DenseLayer(activation_func=Activation.Linear, init_func=init_func,
                       input_size_info=input_size_info, neurons_num=10)
    network.add_layer(layer)

    return network


# Returns loss and accuracy on provided data
def check_performance(network, x, y):
    prediction_list = []
    y_list = []
    # Predict in chunks in case there is a lot of data
    for x_, y_ in full_data_generator(x, y, 10000):
        prediction_list.append(network.predict(x_))
        y_list.append(y_)
    prediction = np.concatenate(prediction_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    acc = Metric.accuracy(prediction, y)
    loss = np.mean(network.loss(prediction, y))
    return acc, loss