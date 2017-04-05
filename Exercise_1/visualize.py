import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Network.history import *


history = History.load("Experiments/Experiment_102_history")
sns.set_style("darkgrid")
print(history.iter)
print(history.valid_acc)

x = np.array(history.iter)
y1 = np.array(history.train_loss)
y2 = np.array(history.valid_loss)
y3 = np.array(history.train_acc)
y4 = np.array(history.valid_acc)

plt.plot(x, y3, label="train accuracy")
plt.plot(x, y4, label="test accuracy")
plt.legend(loc='best', shadow=True)
plt.show()

plt.plot(x, y1, label="train loss")
plt.plot(x, y2, label="test loss")
plt.legend(loc='best', shadow=True)
plt.show()
