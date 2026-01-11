import numpy as np
from scipy import linalg
from ex_2 import X

# 3.
XXt = X @ X.T
val_proprii_XXt, vec_proprii_XXt = linalg.eig(XXt)

XtX = X.T @ X
val_proprii_XtX, vec_proprii_XtX = linalg.eig(XtX)

U, S, Vt = linalg.svd(X)

# valorile proprii lambda ar trebui sa fie sqrt(sigma):
idx_XXt_sorted = np.argsort(val_proprii_XXt.real)[::-1]
lambda_XXt = val_proprii_XXt.real[idx_XXt_sorted]

check = np.allclose(lambda_XXt[:len(S)], S**2)
print(f"Lambda = S^2: {check}")
