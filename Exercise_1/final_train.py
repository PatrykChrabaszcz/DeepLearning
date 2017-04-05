from Network.net import *
from Network.history import *
from Network.data_loader import *


if __name__ == "__main__":
    # Load data
    Dtrain, Dval, Dtest = mnist()
    X_train, y_train = Dtrain
    X_val, y_val = Dval
    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)
    X_test, y_test = Dtest

    experiment_number = 201
    network = Net.load_net(Net.network_filename(100))

    # Saves information about the training
    history = History(solver=network.solver, hidden_size=0, method="Full",
                      batch_size=0)

    # Change the parameters for fine tuning
    network.solver.t = 0
    network.solver.decay_algorithm.lr = 0.5
    network.solver.alpha = 0.0001
    training_samples_num = X_train.shape[0]
    print(training_samples_num)
    epoch_num = 0

    while True:
        for x, y in full_data_generator(X_train, y_train, 10000):
            network.one_step(x, y)
        network.finish_iteration()

        print("\nResults from epoch %i:" % epoch_num)

        # Check train errors
        train_acc, train_loss = network.check_performance(x=X_train, y=y_train)
        print("Acc train: %f" % train_acc)
        print("Loss train: %f" % train_loss)

        # Check validation errors
        test_acc, test_loss = network.check_performance(x=X_test, y=y_test)
        print("Acc valid: %f" % test_acc)
        print("Loss valid: %f" % test_loss)

        history.add_sample(iter=epoch_num, train_acc=train_acc, valid_acc=test_acc,
                           train_loss=train_loss, valid_loss=test_loss)

        network.save_net(Net.network_filename(experiment_number))
        history.save(history_filename(experiment_number))
        epoch_num += 1
        network.solver.advance()
