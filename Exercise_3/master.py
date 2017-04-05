import numpy as np
import time
import signal
end_flag = False


# Used to exit the program after finishing current loop
def signal_handler(signal, frame):
    print('Ending...')
    global end_flag
    end_flag = True


# Generates new set of hyperparameters
# Right now it only generates random settings
# Could be improved to use smarter search techniques
# that use previous evaluations
class HyperParamGenerator:
    def __init__(self):
        self.index = 0
        self.results = []

    # Update internal state based on result
    def update(self, result):
        self.results.append(result)
        with open('solution.txt', 'a') as f:
            f.write(result.line)

    # Generate new hyperparameters based on internal state
    def generate_hyperparams(self):
        # TODO: Use results from previous evaluations
        x = np.random.rand(4)
        num_filters = 10 + int(90 * x[0])  # in [10, 100]
        batch_size = int(pow(2.0, 4.0 + 4.0 * x[1]))  # in [2^4, 2^8] = [16, 256]
        m = float(x[2])  # in [0, 1]
        lr = float(pow(10.0, -2 + 1.5 * x[3]))  # in [10^-2, 10^-0.5] = [0.01, ~0.31]
        algorithm_type = 'SGD'  # SGD with momentum

        self.index += 1
        return '{index}\t{num_filters}\t{batch_size}\t{m}\t{lr}\t{algorithm_type}\n'.\
            format(index=self.index, num_filters=num_filters, batch_size=batch_size,
                   m=m, lr=lr, algorithm_type=algorithm_type)


# Parse infor from result file into this class
class Result:
    def __init__(self):
        self.n_filters = None
        self.batch_size = None
        self.momentum = None
        self.learning_rate = None
        self.algorithm_type = None
        self.accuracy = None
        self.line = None

    def fill_from_line(self, line):
        self.line = line
        print('Received result %s' % self.line)
        # TODO: Add data extraction


# There is no mechanism that will synchronize access to shared files
def main():
    print('Starting ...')
    signal.signal(signal.SIGINT, signal_handler)
    h_generator = HyperParamGenerator()
    result_filename = 'results.txt'
    param_filename = 'hyperparameters.txt'

    buff_size = 1

    while not end_flag:

        # Generate hyperparameters
        with open(param_filename, 'ar+') as f:
            num_lines = len(f.readlines())
            while num_lines < buff_size:
                num_lines += 1
                print('Generating hyperparams')
                f.write(h_generator.generate_hyperparams())

        # Check if there are results present
        success = False
        with open(result_filename, 'ar+') as f:
            lines = f.readlines()
            if lines:
                f.seek(0)
                f.truncate()
                success = True
        if success:
            for l in lines:
                r = Result()
                r.fill_from_line(l)
                h_generator.update(r)

        time.sleep(1)

    print('Finished ...')


if __name__ == '__main__':
    main()

