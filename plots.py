import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
Y = load_digits().target

cdict = ["red", "green", "blue"]


def plot(name):
    with open(name, "r") as f:
        q = f.read().split('\n')

    q = [w.rstrip() for w in q]
    q = [w.split(' ') for w in q]
    q = [list(map(float, w)) for w in q[:-1]]
    q = np.array(q)

    _fig, ax = plt.subplots()
    for g in np.unique(Y):
        ix = np.where(Y == g)
        qq = q[ix]
        ax.scatter(qq[:, 0], qq[:, 1], label=g, s=2)

    ax.legend()
    plt.show()


plot("demo_out")
plot("impl_out")
