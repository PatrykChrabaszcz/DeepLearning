import pickle


def history_filename(i):
    return "Experiments/Experiment_%i_history" % i


class History:
    def __init__(self, solver, hidden_size, method, batch_size):
        self.iter = []
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []
        self.solver = solver
        self.hidden_size = hidden_size
        self.method = method
        self.batch_size = batch_size

    def add_sample(self, iter, train_acc, valid_acc, train_loss, valid_loss):
        self.iter.append(iter)
        self.train_acc.append(train_acc)
        self.valid_acc.append(valid_acc)
        self.train_loss.append(train_loss)
        self.valid_loss.append(valid_loss)

    def clear(self):
        self.iter = []
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []

    def save(self, filename):
        with open(filename, 'wb') as f:
            p = pickle.Pickler(f)
            p.dump(self)

    def __str__(self):
        return "History:\n" +\
               "Solver:\n" +\
               self.solver.__str__() +\
               "hidden_size:\n" +\
               self.hidden_size.__str__() + "\n" +\
               "method: %s\n" % self.method +\
               "batch_size: %i\n" % self.batch_size

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            p = pickle.Unpickler(f)
            return p.load()
