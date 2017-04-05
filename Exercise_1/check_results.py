from Network.data_loader import *
import pprint

from Network.net import *
from Network.history import *

Dtrain, Dval, Dtest = mnist()
X_train, y_train = Dtrain
X_val, y_val = Dval
X_test, y_test = Dtest

# Some problems with memory when for loop is to big
for i in range(102, 103):

    network = Net.load_net(Net.network_filename(i))
    print("Experiment %i:" % i)
    print(network.check_performance(X_test, y_test))

    history = History.load(history_filename(i))
    print(history)
