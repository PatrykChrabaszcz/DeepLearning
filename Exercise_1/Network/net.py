from Network.data_loader import full_data_generator, full_stochastic_generator
import numpy as np
import pickle
import os


class WeightInit:
    # Got to be careful, if weights are too large then it might explode
    @staticmethod
    def gaussian_init(layer):
        layer.w = np.random.normal(scale=0.05, size=layer.w.shape)
        layer.b = np.zeros_like(layer.b)

    # The one which is better for ReLU
    @staticmethod
    def xavier_init_gauss(layer):
        # number of n_in +  number of n_out // Could be also just the number of n_in
        n = layer.w.shape[0] + layer.w.shape[1]
        layer.w = np.random.normal(scale=np.sqrt(2/n), size=layer.w.shape)
        layer.b = np.zeros_like(layer.b)

    @staticmethod
    def xavier_init_uniform(layer):
        # #n_in + #n_out
        n = layer.w.shape[0] + layer.w.shape[1]
        layer.w = np.random.uniform(-np.sqrt(6/n), np.sqrt(6/n), size=layer.w.shape)
        layer.b = np.zeros_like(layer.b)


class Activation:
    class Sigmoid:
        @staticmethod
        def f(x):
            return

        @staticmethod
        def f_d(x):
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (s-1)

    class Tahn:
        @staticmethod
        def f(x):
            return np.tanh(x)

        @staticmethod
        def f_d(x):
            return 1-np.tanh(x)**2

    class Relu:
        @staticmethod
        def f(x):
            return np.maximum(0.0, x)

        @staticmethod
        def f_d(x):
            dx = np.ones_like(x)
            dx[x < 0] = 0
            return dx

    class LeakyRelu:
        @staticmethod
        def f(x):
            x_ = np.copy(x)
            x_[x_ < 0] *= 0.1
            return x_

        @staticmethod
        def f_d(x):
            dx = np.ones_like(x)
            dx[x < 0] = 0.1
            return dx

    class Linear:
        @staticmethod
        def f(x):
            return x

        @staticmethod
        def f_d(x):
            return 1

    class Softmax:
        @staticmethod
        def f(x):
            max = np.max(x, axis=1, keepdims=True)
            x_n = x - max

            exp = np.exp(x_n)
            return exp / np.sum(exp, axis=1, keepdims=True)

        # Softmax derivative is a little bit more complex (Jacobian matrix)
        # It's better to use CrossEntropyLoss with logits because some
        # equations will simplify
        @staticmethod
        def f_d(x):
            raise NotImplementedError


# In decay(self, t) 't' should start from 0
class LRDecay:
    # After constant number of steps('step') decay by constant factor('rate')
    class StepDecay:
        def __init__(self, learning_rate, step, rate):
            self.lr = learning_rate
            self.step = step
            self.rate = rate

        def learning_rate(self, t):
            f = int(t/self.step)
            return self.lr*(self.rate**f)

        def __str__(self):
            return "StepDecay\n" +\
                   "step: %i \n" % self.step +\
                   "rate: %f \n" % self.rate
#                   "learning_rate: %f \n" % self.lr +\

    # SGDR
    # Never reaches learning_rate_min because 'restart' is performed earlier
    # Updates only once for an epoch (no updates between mini batches)
    class SimulatedRestarts:
        def __init__(self, learning_rate_max, learning_rate_min, t_0, t_mul):
            self.n_min = learning_rate_min
            self.n_max = learning_rate_max
            self.t_0 = t_0
            self.t_mul = t_mul

        def learning_rate(self, t):
            t_curr = self.t_0
            while t >= t_curr:
                t -= t_curr
                t_curr *= self.t_mul

            return self.n_min + 0.5*(self.n_max - self.n_min)*(1 + np.cos(t*np.pi/t_curr))

        def __str__(self):
            return "Simulated Restarts: \n" +\
                   "learning_rate_max: %f \n" % self.n_max +\
                   "learning_rate_min: %f \n" % self.n_min +\
                   "t_0: %i \n" % self.t_0 +\
                   "t_mul: %i \n" % self.t_mul


class Solver:
    class Base:
        def __init__(self):
            self.network = None
            self.t = 0

        # Has to be called after all layers are created
        def set_network(self, network):
            self.network = network

        def update_layer(self, layer):
            w_update, b_update = self.compute_update(layer)
            layer.update(w_update, b_update)

        def advance(self):
            self.t += 1

        def compute_update(self, layer):
            raise NotImplementedError

    class Simple(Base):
        def __init__(self, decay_algorithm, alpha=0):
            super().__init__()
            self.decay_algorithm = decay_algorithm
            self.alpha = alpha

        def compute_update(self, layer):
            learning_rate = self.decay_algorithm.learning_rate(self.t)
            w_update = -learning_rate * (layer.dw + self.alpha*layer.w)
            b_update = -learning_rate * (layer.db + self.alpha*layer.b)
            return w_update, b_update

        def __str__(self):
            return "Solver Simple \n" + \
                   "Alpha: %f \n" % self.alpha + \
                   "decay_algorithm: \n" + \
                   self.decay_algorithm.__str__()

    class Momentum(Base):
        def __init__(self, decay_algorithm, momentum_rate, alpha=0):
            super().__init__()
            self.decay_algorithm = decay_algorithm
            self.momentum_rate = momentum_rate
            self.momentum = 0
            self.alpha = alpha

        def set_network(self, network):
            self.network = network
            for layer in self.network.layers:
                layer.momentum_w = np.zeros_like(layer.w, dtype='float32')
                layer.momentum_b = np.zeros_like(layer.b, dtype='float32')

        def compute_update(self, layer):
            learning_rate = self.decay_algorithm.learning_rate(self.t)
            momentum_w = self.momentum_rate * layer.momentum_w - \
                         learning_rate * (layer.dw + self.alpha * layer.w)
            momentum_b = self.momentum_rate * layer.momentum_b - \
                         learning_rate * (layer.db + self.alpha * layer.b)
            return momentum_w, momentum_b

        # Add new attribute to the layer class
        def advance(self):
            learning_rate = self.decay_algorithm.learning_rate(self.t)
            for layer in self.network.layers:
                layer.momentum_w = self.momentum_rate * layer.momentum_w - \
                                   learning_rate * (layer.dw + self.alpha * layer.w)
                layer.momentum_b = self.momentum_rate * layer.momentum_b - \
                                   learning_rate * (layer.db + self.alpha * layer.b)
            self.t += 1

        def __str__(self):
            return "Solver Momentum \n" + \
                   "Momentum Rate: %f \n" % self.momentum_rate + \
                   "Alpha: %f \n" % self.alpha + \
                   "decay_algorithm: \n" + \
                   self.decay_algorithm.__str__()


# Loss is Nx1 np.array ('N' is num of samples in the batch)
# f(pred, targ) - Loss function
# f_d(pred, targ) - Derivative of loss w.r.t. pred
class Objective:
    class Squared:
        @staticmethod
        def loss(pred, targ):
            return np.array(0.5 * (pred - targ) ** 2)

        @staticmethod
        def loss_d(pred, targ):
            return np.array(pred - targ)

    class CrossEntropyWithLogits:
        @staticmethod
        def loss(pred, targ):
            # If prediction is 0 we have numerical problems

            # Subtracting a constant from each prediction does not change the output
            # but improves the numerical stability
            max_pred = np.max(pred, axis=1, keepdims=True)
            pred = pred - max_pred

            exp = np.exp(pred)
            probabilities = exp / np.sum(exp, axis=1, keepdims=True)
            probabilities = np.maximum(probabilities, 1e-15)

            return np.mean(-targ*np.log(probabilities), axis=1)

        @staticmethod
        def loss_d(pred, targ):
            return pred - targ


class Metric:
    @staticmethod
    def accuracy(pred, targ):
        p = np.argmax(pred, axis=1)
        t = np.argmax(targ, axis=1)
        acc = np.sum(p == t)/len(p)
        return acc


class Net(object):
    def __init__(self, solver, objective):
        self.layers = []
        self.solver = solver
        self.objective = objective

    def add_layer(self, layer):
        self.layers.append(layer)

    # One way to train is to call one_interation()
    # Weights will be updated after a single batch
    def one_iteration(self, x, y):
        p = self.predict(x)
        self._backward_pass(p, y)
        self._update()

    # Other way to train is to call one_step() until
    # all data is forwarded, then call finish_iteration()
    def one_step(self, x, y):
        p = self.predict(x)
        self._backward_pass(p, y)

    def finish_iteration(self):
        self._update()

    # Forward pass through all layers
    def predict(self, x):
        p = self.layers[0].forward_pass(x)
        for l in self.layers[1:]:
            p = l.forward_pass(p)
        return p

    # Backward pass through all layers
    def _backward_pass(self, p, y):
        dl = self.objective.loss_d(pred=p, targ=y)
        for l in self.layers[::-1]:
            dl = l.backward_pass(dl)

    def _update(self):
        for l in self.layers:
            l.average_gradients()
            self.solver.update_layer(l)

    # Returns loss and accuracy on provided data
    def check_performance(self, x, y):
        prediction_list = []
        y_list = []
        # Predict in chunks in case there is a lot of data
        for x_, y_ in full_data_generator(x, y, 10000):
            prediction_list.append(self.predict(x_))
            y_list.append(y_)
        prediction = np.concatenate(prediction_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        acc = Metric.accuracy(prediction, y)
        loss = np.mean(self.loss(prediction, y))
        return acc, loss

    def loss(self, p, y):
        return self.objective.loss(pred=p, targ=y)

    def save_net(self, filename):
        path, file = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
        with open(filename, 'wb') as f:
            p = pickle.Pickler(f)
            p.dump(self)

    @staticmethod
    def network_filename(i):
        return "Experiments/Experiment_%i_network" % i

    @staticmethod
    def load_net(filename):
        with open(filename, 'rb') as f:
            p = pickle.Unpickler(f)
            return p.load()


class DenseLayer(object):
    def __init__(self, activation, input_size_info=None, init_func=WeightInit.xavier_init_gauss,
                 neurons_num=100):
        try:
            input_size = input_size_info.neurons_num
        except AttributeError:
            input_size = input_size_info

        self.neurons_num = neurons_num
        self.w = np.zeros(shape=(input_size, neurons_num), dtype='float32')
        self.b = np.zeros(shape=(1, neurons_num), dtype='float32')
        init_func(self)

        self.x = None                       # Input
        self.dx = None                      # Gradient w.r.t. input (List for data chunks)
        self.dw = np.zeros_like(self.w)     # Gradient w.r.t. weights (List for data chunks)
        self.db = np.zeros_like(self.b)     # Gradient w.r.t. biases (List for data chunks)

        # Lists used for gradient averaging
        self.dx_list = []
        self.dw_list = []
        self.db_list = []
        self.s_list = []
        self.a = None   # x @ w
        self.y = None   # Output
        self.activation = activation

    def forward_pass(self, x):
        self.x = x
        self.a = x @ self.w + self.b
        self.y = self.activation.f(self.a)
        return self.y

    def backward_pass(self, dl):
        da = dl * self.activation.f_d(self.a)
        self.db = da.mean(axis=0)
        self.dw = self.x.T @ da / self.x.shape[0]
        self.dx = da @ self.w.T
        s = self.x.shape[0]

        # Store gradients from batches until next update is called
        self.db_list.append(self.db)
        self.dw_list.append(self.dw)
        self.dx_list.append(self.dx)
        self.s_list.append(s)

        return self.dx

    def average_gradients(self):
        # Not sure if it's nice that self.dw type changes from list to scalar
        s = self.s_list/np.sum(self.s_list)
        self.dw = np.average(a=self.dw_list, axis=0, weights=s)
        # self.db has to be the same shape as self.b, it has to be 2 dimensional
        self.db = np.array([np.average(a=self.db_list, axis=0, weights=s)])

    def update(self, dw, db):
        self.w += dw
        self.b += db

        # Clear lists with gradients
        self.db_list = []
        self.dw_list = []
        self.dx_list = []
        self.s_list = []
