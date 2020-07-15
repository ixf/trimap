from sklearn.datasets import load_digits
import numpy as np
X = load_digits().data

X -= np.min(X)
X /= np.max(X)
X -= np.mean(X, axis=0)

with open("mnist_data", "w") as f:
    for row in X:
        s = ' '.join(map(str, row))
        f.write(s)
        f.write("\n")