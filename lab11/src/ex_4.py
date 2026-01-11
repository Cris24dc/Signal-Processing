import numpy as np
from scipy import linalg
from ex_1 import time_series_y
from ex_2 import X, L

# 4.
def SSA(matrix):
    L, K = matrix.shape
    x = np.zeros(L+K-1)
    
    for k in range(L+K-1):
        values = []

        for i in range(L):
            j = k-i
            if 0 <= j < K:
                values.append(matrix[i, j])

        x[k] = np.mean(values)
        
    return x


U, S, Vt = linalg.svd(X)
x_hat = []

for i in range(L):
    X_i = S[i] * np.outer(U[:,i], Vt[i,:])
    
    x_i_hat = SSA(X_i)
    x_hat.append(x_i_hat)

x_hat = np.array(x_hat)
x = np.sum(x_hat, axis=0)

check = np.allclose(time_series_y, x)
print(f"Descompus corect: {check}")