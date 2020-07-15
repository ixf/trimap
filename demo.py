import trimap
from sklearn.datasets import load_digits

digits = load_digits()
embedding = trimap.TRIMAP().fit_transform(digits.data)

with open("demo_out", "w") as f:
    for row in embedding:
        s = ' '.join(map(str, row))
        f.write(s)
        f.write("\n")
