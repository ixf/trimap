import numpy as np
import matplotlib.pyplot as plt

with open("impl_out", "r") as f:
    q = f.read().split('\n')

q = [w.rstrip() for w in q]
q = [w.split(' ') for w in q]
q = [list(map(float, w)) for w in q[:-1]]
q = np.array(q)

plt.scatter(q[:, 0], q[:, 1])

plt.show()

with open("demo_out", "r") as f:
    q = f.read().split('\n')

q = [w.rstrip() for w in q]
q = [w.split(' ') for w in q]
q = [list(map(float, w)) for w in q[:-1]]
q = np.array(q)

plt.scatter(q[:, 0], q[:, 1])
plt.show()
