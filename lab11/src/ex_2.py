import numpy as np
from ex_1 import time_series_y, N

# 2.
L = 20
K = N - L + 1
X = np.zeros((L, K))

for i in range(L):
    X[i] = time_series_y[i:i+K]

if __name__ == "__main__":
    print(X)