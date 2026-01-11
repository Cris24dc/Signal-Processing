import numpy as np

# 3.
def find_roots(coef):
    N = len(coef)-1
    C = np.zeros((N, N))

    coef = np.array(coef, dtype=float)
    coef = coef[1:]/coef[0]
    
    C[np.arange(1, N), np.arange(1, N)-1] = 1
    C[:, -1] = -coef[::-1]
    
    return np.linalg.eigvals(C)
