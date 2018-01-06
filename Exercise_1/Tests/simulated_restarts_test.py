import seaborn as sns
import matplotlib.pyplot as plt
from Network.net import *

# Plot learning rate over time
if __name__ == "__main__":
    sns.set_style("darkgrid")

    decay = LRDecay.SimulatedRestarts(learning_rate_max=0.5, learning_rate_min=0, t_0=5, t_mul=2)

    t = list(range(100))
    y = []

    for i in t:
        y.append(decay.learning_rate(i))

    plt.plot(t, y)
    plt.show()
