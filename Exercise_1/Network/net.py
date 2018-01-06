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
        def f_grad(x):
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (s-1)

    class Tahn:
        @staticmethod
        def f(x):
            return np.tanh(x)

        @staticmethod
        def f_grad(x):
            return 1-np.tanh(x)**2

    class Relu:
        @staticmethod
        def f(x):
            return np.maximum(0.0, x)

        @staticmethod
        def f_grad(x):
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
        def f_grad(x):
            dx = np.ones_like(x)
            dx[x < 0] = 0.1
            return dx

    class Linear:
        @staticmethod
        def f(x):
            return x

        @staticmethod
        def f_grad(x):
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
        def f_grad(x):
            raise NotImplementedError


# In decay(self, t) 't' should start from 0
class LRDecay:

    class NoDecay:
        def __init__(self, learning_rate):
            self.lr = learning_rate

        def learning_rate(self, t):
            return self.lr

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
            self.n_max = learning_rate_max
            self.n_min = learning_rate_min
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
        def __init__(self, network):
            self.network = network
            self.t = 0

        def one_step(self, x, y):
            p = self.network.predict(x)
            self.network.backward_pass(p, y)

        def finish_iteration(self):
            self._update()

        def _update(self):
            for l in self.network.layers:
                delta_w, delta_b = self._compute_update(l)
                l.update(delta_w, delta_b)
            self._advance()

        def _advance(self):
            self.t += 1

        def _compute_update(self, layer):
            raise NotImplementedError

    class Simple(Base):
        def __init__(self, network, decay_algorithm, alpha=0.0):
            super().__init__(network)
            self.decay_algorithm = decay_algorithm
            self.alpha = alpha

        def _compute_update(self, layer):
            lr = self.decay_algorithm.learning_rate(self.t)
            grad_w = layer.grad_w_acc.mean_gradient()
            grad_b = layer.grad_b_acc.mean_gradient()
            delta_w = -lr * (grad_w + self.alpha*layer.w)
            delta_b = -lr * grad_b

            return delta_w, delta_b

        def __str__(self):
            return "Solver Simple \n" + \
                   "Alpha: %f \n" % self.alpha + \
                   "decay_algorithm: \n" + \
                   self.decay_algorithm.__str__()

    class Momentum(Base):
        def __init__(self, network, decay_algorithm, momentum_rate=0.9, alpha=0.0):
            super().__init__(network)
            self.decay_algorithm = decay_algorithm
            self.momentum_rate = momentum_rate
            self.alpha = alpha
            self.network = network

            # We store momentum values inside the layer objects
            # Maybe it's better to keep them inside the optimizer ?
            for layer in self.network.layers:
                layer.momentum_w = np.zeros_like(layer.w, dtype='float32')
                layer.momentum_b = np.zeros_like(layer.b, dtype='float32')

        def _compute_update(self, layer):
            lr = self.decay_algorithm.learning_rate(self.t)
            grad_w = layer.grad_w_acc.mean_gradient()
            grad_b = layer.grad_b_acc.mean_gradient()
            layer.momentum_w = self.momentum_rate * layer.momentum_w - lr * (grad_w + self.alpha * layer.w)
            layer.momentum_b = self.momentum_rate * layer.momentum_b - lr * grad_b

            return layer.momentum_w, layer.momentum_b

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


# Accumulate gradients from multiple mini-batches
# Computes the mean gradient
class GradientAccumulator:
    def __init__(self):
        self.gradient = None
        self.cnt = 0

    def append(self, gradient):
        self.gradient = gradient if self.gradient is None else self.gradient + gradient
        self.cnt += 1

    def mean_gradient(self):
        return self.gradient / self.cnt

    def reset(self):
        self.gradient = None
        self.cnt = 0


class Net(object):
    def __init__(self, objective):
        self.layers = []
        self.objective = objective

    def add_layer(self, layer):
        self.layers.append(layer)

    # Forward pass through all layers
    def predict(self, x):
        p = self.layers[0].forward_pass(x)
        for l in self.layers[1:]:
            p = l.forward_pass(p)
        return p

    # Backward pass through all layers
    def backward_pass(self, p, y):
        grad_loss = self.objective.loss_d(pred=p, targ=y)
        for l in self.layers[::-1]:
            grad_loss = l.backward_pass(grad_loss)

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
    def __init__(self, activation_func, input_size_info=None, init_func=WeightInit.xavier_init_gauss,
                 neurons_num=100):
        try:
            input_size = input_size_info.neurons_num
        except AttributeError:
            input_size = input_size_info

        self.neurons_num = neurons_num
        self.activation_func = activation_func

        self.w = np.zeros(shape=(input_size, neurons_num), dtype='float32')
        self.b = np.zeros(shape=(1, neurons_num), dtype='float32')
        init_func(self)

        # In some settings we want to use the mean gradient from
        # multiple mini-batches or even from the whole dataset.
        self.grad_w_acc = GradientAccumulator()
        self.grad_b_acc = GradientAccumulator()

        self.x = None   # Input x
        self.a = None   # x @ w
        self.y = None   # Output

    def forward_pass(self, x):
        self.x = x
        self.a = x @ self.w + self.b
        self.y = self.activation_func.f(self.a)
        return self.y

    def backward_pass(self, grad_loss):
        grad_a = grad_loss * self.activation_func.f_grad(self.a)
        grad_b = grad_a.mean(axis=0)
        grad_w = self.x.T @ grad_a / self.x.shape[0]
        grad_x = grad_a @ self.w.T

        # Store gradients from batches until next update is called
        self.grad_b_acc.append(grad_b)
        self.grad_w_acc.append(grad_w)

        return grad_x

    def update(self, delta_w, delta_b):
        self.w += delta_w
        self.b += delta_b
        self.grad_w_acc.reset()
        self.grad_b_acc.reset()
