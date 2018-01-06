from Network.net import *
from Network.history import *
from Network.data_loader import *
import click
from utils import create_network, check_performance, Stats

# Script to train network with user user defined parameters

# We have 10 classes in MNIST dataset
NUM_CLASSES = 10


@click.command()
@click.option('--training_method', type=click.Choice(['epoch_gd', 'epoch_sgd', 'random_sgd']), default='epoch_sgd',
              help='epoch_gd: One weight update after processing all data from the dataset.\n'
                   'epoch_sgd: One weight update after each mini-batch of data, data shuffled after each epoch.\n'
                   'random_sgd: One weight update after each mini-batch of data, data shuffled after each iteration.')
@click.option('--experiment_number', type=int, default=100)
@click.option('--batch_size', type=int, default=128)
def main(training_method, experiment_number, batch_size):
    # Load the data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist()

    x_train = np.concatenate([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])

    # lr_decay = LRDecay.SimulatedRestarts(learning_rate_max=learning_rate_max,
    #                                      learning_rate_min=learning_rate_min,
    #                                      t_0=t_0, t_mul=t_mul)

    lr_decay = LRDecay.StepDecay(learning_rate=0.5, step=5000, rate=0.5)

    hidden_size = (900,)
    network = create_network(hidden_size=hidden_size)

    solver = Solver.Simple(network, decay_algorithm=lr_decay, alpha=0.0)

    # Saves information about the training
    history = History(solver=solver, hidden_size=hidden_size, method=training_method,
                      batch_size=batch_size)

    training_samples_num = x_train.shape[0]

    epoch_num = 0
    best_test_loss = np.inf

    main_stats = Stats('Main Stats')
    train_stats = main_stats.create_child_stats('Train Procedure')
    train_v_stats = main_stats.create_child_stats('Train Set Validation')
    test_v_stats = main_stats.create_child_stats('Test Set Validation')
    saver_stats = main_stats.create_child_stats('Saver')
    while True:
        with main_stats:
            with train_stats:
                # Full gradient descent learning
                if training_method == 'epoch_gd':
                    for x, y in full_data_generator(x_train, y_train, 10000):
                        solver.one_step(x, y)
                    solver.finish_iteration()

                # Stochastic learning (shuffled input data)
                elif training_method == 'epoch_sgd':
                    for x, y in full_stochastic_generator(x_train, y_train, batch_size):
                        solver.one_step(x, y)
                        solver.finish_iteration()

                # Stochastic learning (random batch)
                elif training_method == 'random_sgd':
                    for j in range(int(training_samples_num/batch_size)):
                        x, y = stochastic_data(x_train, y_train, batch_size=batch_size)
                        solver.one_step(x, y)
                        solver.finish_iteration()

                print("\nResults from epoch %i:" % epoch_num)
                print("Current learning rate %f" % lr_decay.learning_rate(solver.t))

            with train_v_stats:
                # Check train errors
                train_acc, train_loss = check_performance(network, x=x_train, y=y_train)
                print("Acc train: %f" % train_acc)
                print("Loss train: %f" % train_loss)

            with test_v_stats:
                # Check validation errors
                test_acc, test_loss = check_performance(network, x=x_test, y=y_test)
                print("Acc test: %f" % test_acc)
                print("Loss test: %f" % test_loss)

            with saver_stats:
                history.add_sample(iter=epoch_num, train_acc=train_acc, valid_acc=test_acc,
                                   train_loss=train_loss, valid_loss=test_loss)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    #network.save_net(Net.network_filename(experiment_number))

                epoch_num += 1

                #history.save(history_filename(experiment_number))


if __name__ == "__main__":
    main()
