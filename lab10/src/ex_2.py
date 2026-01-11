import numpy as np
import matplotlib.pyplot as plt
from ex_1 import time, time_series_y

# 2.
def AR(x, p):
    N = len(x)
    m = np.mean(x)
    centered_x = x - m
    params = []
    targets = []
    
    for t in range(p, N):
        lags = centered_x[t-p:t][::-1]
        params.append(lags)
        targets.append(centered_x[t])
    
    phi, _, _, _ = np.linalg.lstsq(np.array(params), np.array(targets), rcond=None)
    
    predictions = np.zeros(N)
    predictions[:p] = x[:p]
    
    for t in range(p, N):
        lags = centered_x[t-p:t][::-1]
        predictions[t] = m + np.dot(phi, lags)
        
    return predictions, phi

p = 50
predictions, _ = AR(time_series_y, p)
mse = np.mean((time_series_y[p:] - predictions[p:])**2)

colors = ['darkorange', 'forestgreen']
titles = ['Original Time Series', f'AR Model (p={p}, MSE={mse:.2f})']
Ox = [time] * 2
Oy = [time_series_y, predictions]

fig, axs = plt.subplots(2, 1, figsize=(12, 10))

for i in range(2):
    axs[i].plot(Ox[i], Oy[i], color=colors[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Amplitude')
    axs[i].grid()

plt.tight_layout()
plt.savefig('./img/Ex_2.pdf', format='pdf')
