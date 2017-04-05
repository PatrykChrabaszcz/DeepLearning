from argparse import ArgumentParser
import signal
import time
from base import *
import theano.tensor as T
import theano


end_flag = False


def signal_handler(signal, frame):
    global end_flag
    end_flag = True


def main(task, algorithm):
    np.set_printoptions(threshold=np.nan)
    net_filename = 'net_{}_{}_'.format(task, algorithm)
    stat_filename = 'stat_{}_{}'.format(task, algorithm)
    signal.signal(signal.SIGINT, signal_handler)

    print("Loading data...")
    data = load_data()
    print("Train data shape: ", data['X_train'].shape)
    print("Train label shape: ", data['Y_train'].shape)
    print("Validation data shape: ", data['X_val'].shape)
    print("Validation label shape: ", data['Y_val'].shape)
    print("Test data shape: ", data['X_test'].shape)
    print("Test label shape: ", data['Y_test'].shape)

    print("Building model...")

    X_train = data['X_train']
    X_val = data['X_val']
    input = T.tensor4("input")
    target = T.matrix("target")
    if task == 'Gender':
	acc_index = 0
        attr_index = data['attr_names'].index('Male')
        n_out = 1
        y_train = data['Y_train'][:, attr_index, None]
        y_val = data['Y_val'][:, attr_index, None]
    if task == 'All':
	acc_index = data['attr_names'].index('Male')
        n_out = 40
        y_train = data['Y_train']
        y_val = data['Y_val']

    network = create_network(input_var=input, n_out=n_out)

    print("Number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    alpha = theano.shared(lasagne.utils.floatX(0.001))
    lr = theano.shared(lasagne.utils.floatX(0.01))

    # Train
    train_prediction = lasagne.layers.get_output(network)
    train_prediction = theano.tensor.clip(train_prediction, 0.001, 0.999)
    train_loss = lasagne.objectives.binary_crossentropy(predictions=train_prediction, targets=target)
    train_loss = lasagne.objectives.aggregate(train_loss)

    l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    train_loss += alpha * l2_penalty

    # Solver
    params = lasagne.layers.get_all_params(network, trainable=True)
    if algorithm == 'Nestrov':
        updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=lr, momentum=0.9)
    if algorithm == "Adam":
	lr.set_value(lasagne.utils.floatX(0.001))
        updates = lasagne.updates.adam(train_loss, params, learning_rate=lr)

    # Test
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(predictions=test_prediction, targets=target)
    test_loss = lasagne.objectives.aggregate(test_loss)
    test_loss += alpha * l2_penalty

    train_acc = lasagne.objectives.binary_accuracy(predictions=train_prediction, targets=target)
    test_acc = lasagne.objectives.binary_accuracy(predictions=test_prediction, targets=target)

    print("Compiling functions...")
    train_function = theano.function([input, target], [train_loss, train_acc], updates=updates)
    validation_function = theano.function([input, target], [test_loss, test_acc])

    print("Starting training...")

    epoch_num = 0
    best_loss = np.inf
    best_val_acc = 0

    file = open(stat_filename, 'w+', 0)
    start_time = time.time()

    while not end_flag:
        print("Starting epoch %i" % epoch_num)
        epoch_start_time = time.time()
        count = 0
	    acc_sum = 0
	    loss_sum = 0

        for x, y in batch_generator(X_train, y_train, batch_size=256, shuffle=True):
	    loss, acc = train_function(x, y)
            acc_sum += acc[:,acc_index, None].mean()
            loss_sum += loss
            count += 1
        train_loss_mean = loss_sum/count
        train_acc_mean = acc_sum/count

        count = 0
        acc_sum = 0
        loss_sum = 0
        for x, y in batch_generator(X_val, y_val,
                                    batch_size=256, shuffle=False):
            loss, acc = validation_function(x, y)
            acc_sum += acc[:, acc_index, None].mean()
            loss_sum += loss
            count += 1
        val_loss_mean = loss_sum/count
        val_acc_mean = acc_sum/count
        print("Epoch took: %f seconds" % (time.time() - epoch_start_time))
        print("Val loss: %f \t\t| Val acc: %f" % (val_loss_mean, val_acc_mean))
        print("Train loss: %f\t\t| Train acc: %f" % (train_loss_mean, train_acc_mean))

        file.write("{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(epoch_num, time.time()-start_time,
                                                                   val_loss_mean, val_acc_mean, train_loss_mean))

	if not epoch_num % 5:
		save_net_params(network, net_filename + str(epoch_num))

	if not epoch_num % 50 and epoch_num != 0:
	    lr.set_value(lasagne.utils.floatX(lr.get_value()/2))

        if val_loss_mean < best_loss:
            best_loss = val_loss_mean
            #save_net_params(network, net_filename)

        if best_val_acc < val_acc_mean:
            best_val_acc = val_acc_mean

        epoch_num += 1

    print('Best val acc: %f' % best_val_acc)
    file.close()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-s', '--scenario', help="Scenario for the scrip, see README")
    args = parser.parse_args()
    return args.scenario


if __name__ == '__main__':
    scenario = parse_arguments()

    if scenario == '1':
        main(task='Gender', algorithm='Nestrov')
    if scenario == '2':
        main(task='Gender', algorithm='Adam')
    if scenario == '3':
        main(task='All', algorithm='Nestrov')
    if scenario == '4':
        main(task='All', algorithm='Adam')

