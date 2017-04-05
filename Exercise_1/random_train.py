from Network.net import *
from Network.history import *
from Network.data_loader import *
import time
import random


# Script to train networks on randomly selected parameters.

# Parameters update after whole batch is processed
FULL_BATCH = 1
# Shuffle the data at the beginning and then return chunks
# and update parameters after each chunk is processed.
STOCHASTIC_FULL = 2
# Almost like STOCHASTIC_FULL but data is shuffled for each batch
# So not all samples are processed in each epoch, and some samples
# can be in many batches in one epoch
STOCHASTIC = 3

method_name = ['', 'FULL_BATCH', 'STOCHASTIC_FULL', 'STOCHASTIC']


# hidden_size list with number of neurons in each layer
# doesn't include output and input layers
def create_network(solver, hidden_size=[], alpha=0):

    network = Net(solver=solver, objective=Objective.SoftmaxMaxLikelihood)

    # init_func = WeightInit.gaussian_init
    init_func = WeightInit.xavier_init_gauss

    input_size_info = 784
    for size in hidden_size:
        layer = DenseLayer(activation=Activation.Relu, init_func=init_func,
                           input_size_info=input_size_info, neurons_num=size)
        network.add_layer(layer)
        input_size_info = layer

    layer = DenseLayer(activation=Activation.Softmax, init_func=init_func,
                       input_size_info=layer, neurons_num=10)
    network.add_layer(layer)

    solver.set_network(network)

    return network


if __name__ == "__main__":
    # Load data
    Dtrain, Dval, Dtest = mnist()
    X_train, y_train = Dtrain
    X_val, y_val = Dval
    X_test, y_test = Dtest

    experiment_number = 0
    # Do experiments until user sends stop signal
    while True:
        start_time = time.time()
        experiment_number += 1

        decayType = random.choice(["SimulatedRestarts", "Step"])
        if decayType == "SimulatedRestarts":
            # 0.1 to 3.16
            learning_rate_max = 10**random.uniform(-1, 0.5)
            learning_rate_min = 0
            t_0 = int(random.uniform(10, 50))
            # 1 or 2
            t_mul = int(random.uniform(1, 3))
            lr_decay = LRDecay.SimulatedRestarts(learning_rate_max=learning_rate_max,
                                                 learning_rate_min=learning_rate_min,
                                                 t_0=t_0, t_mul=t_mul)

        elif decayType == "Step":
            # 0.1 to 3.16
            learning_rate = 10**random.uniform(-1, 1)
            step = int(random.uniform(10, 50))
            rate = random.uniform(0.2, 1)
            lr_decay = LRDecay.StepDecay(learning_rate=learning_rate, step=step, rate=rate)

        solverType = random.choice(["Simple", "Momentum"])

        # Simple could be just momentum with 0 momentum
        # But not 100% sure about momentum implementation
        if solverType == "Simple":
            solver = Solver.Simple(decay_algorithm=lr_decay)
        elif solverType == "Momentum":
            momentum_rate = random.uniform(0, 1)
            solver = Solver.Momentum(decay_algorithm=lr_decay, momentum_rate=momentum_rate)

        print("Solver for this experiment:")
        print(solver)

        hidden_size = []

        # 50% chance that we will add another layer
        temp = [True, False]
        add_layer = True
        while add_layer:
            hidden_size.append(int(random.uniform(150, 800)))
            add_layer = random.choice(temp)

        #from 0.0000001 to 0.1
        alpha = 10 ** random.uniform(-7, -1)

        print("Hidden Size:")
        print(hidden_size)

        # 33% that it will be reset to 0
        alpha = random.choice([0, alpha, alpha])
        print("Alpha: %f \n" % alpha)

        network = create_network(solver=solver, hidden_size=hidden_size, alpha=alpha)

        # Choose learning method
        learning_method = random.choice([FULL_BATCH, STOCHASTIC_FULL, STOCHASTIC])
        batch_size = random.choice([64, 128, 256, 512, 1024, 2048, 4096])
        print("learning method: %s" % method_name[learning_method])
        if learning_method != FULL_BATCH:
            print("batch size: %i" % batch_size)

        # Saves information about the training
        history = History(solver=network.solver, hidden_size=hidden_size, method=method_name[learning_method],
                          batch_size=batch_size)

        training_samples_num = X_train.shape[0]

        epoch_num = 0
        best_val_loss = np.inf
        # Give 40 minutes to train each network
        while time.time()-start_time < 2400:

            # Full batch learning
            if learning_method == FULL_BATCH:
                for x, y in full_data_generator(X_train, y_train, 10000):
                    network.one_step(x, y)
                network.finish_iteration()

            # Stochastic learning (random batch)
            elif learning_method == STOCHASTIC:
                # It should not make a big difference if we don't use
                # this last n=(training_samples % batch_size) samples
                for j in range(int(training_samples_num/batch_size)):
                    x, y = stochastic_data(X_train, y_train, batch_size)
                    network.one_iteration(x, y)
                    # Call to solver.advance() either here or at the end of the epoch
                    # solver.advance()

            # Stochastic learning (shuffled input data)
            elif learning_method == STOCHASTIC_FULL:
                for x, y in full_stochastic_generator(X_train, y_train, batch_size):
                    network.one_iteration(x, y)

            print("\nResults from epoch %i:" % epoch_num)
            print("Current learning rate %f" % lr_decay.learning_rate(solver.t))

            # Check train errors
            train_acc, train_loss = network.check_performance(x=X_train, y=y_train)
            print("Acc train: %f" % train_acc)
            print("Loss train: %f" % train_loss)

            # Check validation errors
            val_acc, val_loss = network.check_performance(x=X_val, y=y_val)
            print("Acc valid: %f" % val_acc)
            print("Loss valid: %f" % val_loss)

            history.add_sample(iter=epoch_num, train_acc=train_acc, valid_acc=val_acc,
                               train_loss=train_loss, valid_loss=val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                network.save_net(Net.network_filename(experiment_number))

            solver.advance()
            epoch_num += 1

        # Save history after experiment is finished
        history.save(history_filename(experiment_number))
