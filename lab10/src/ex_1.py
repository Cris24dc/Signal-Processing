import numpy as np

# 1.
N = 1000
time = np.arange(N)

trend_y = 0.00001*time**2 + 0.0003*time + 1
seasonal_y = 5 * (1/2*np.cos(2/50*np.pi*time + 1) + 1/5*np.cos(2/80*np.pi*time + 3))
residuals_y = np.random.normal(0, 1, N)
time_series_y = trend_y + seasonal_y + residuals_y
