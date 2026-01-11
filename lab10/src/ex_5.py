import numpy as np
from ex_1 import time_series_y
from ex_2 import AR
from ex_4 import find_roots

p = 50
_, coef = AR(time_series_y, p)
coefs = np.concatenate((-coef[::-1], [1]))

magnitudes = np.abs(find_roots(coefs))
is_stationary = np.all(magnitudes > 1)

print(f"IS stationary? {'YES' if is_stationary else 'NO'}")
