from Network.net import *
from Network.history import *
from Network.data_loader import *

# Script to train network with user defined parameters

# Parameters update after whole batch is processed
FULL_BATCH = 1
# Shuffle the data at the beginning and then return chunks
# and update parameters after each chunk is processed.
# Advance solver after epoch.
STOCHASTIC_FULL = 2
# Take random samples from the data and update parameters
# after those samples are processed.
# Advance solver after one batch.
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
    X_train = np.concatenate([X_train, X_val])
    y_train = np.concatenate([y_train, y_val])
    X_val = X_test
    y_val = y_test

    experiment_number = 103

    # lr_decay = LRDecay.SimulatedRestarts(learning_rate_max=learning_rate_max,
    #                                      learning_rate_min=learning_rate_min,
    #                                      t_0=t_0, t_mul=t_mul)

    lr_decay = LRDecay.StepDecay(learning_rate=0.75, step=20, rate=0.5)

    solver = Solver.Simple(decay_algorithm=lr_decay)
    # solver = Solver.Momentum(decay_algorithm=lr_decay, momentum_rate=0.9)

    hidden_size = [900]
    learning_method = STOCHASTIC
    batch_size = 512
    network = create_network(solver=solver, hidden_size=hidden_size, alpha=0.00001)

    # Saves information about the training
    history = History(solver=network.solver, hidden_size=hidden_size, method=method_name[learning_method],
                      batch_size=batch_size)

    training_samples_num = X_train.shape[0]

    epoch_num = 0
    best_val_loss = np.inf

    while True:

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
                x, y = stochastic_data(X_train, y_train, batch_size=batch_size)
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

        solver.advance()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            network.save_net(Net.network_filename(experiment_number))

        epoch_num += 1

        history.save(history_filename(experiment_number))

