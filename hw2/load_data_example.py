import numpy as np
from pystruct.datasets import load_letters
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more convenient
X, y = np.array(X), np.array(y)

print X.shape
print y.shape

for i in range(1000):
    print y[i]
