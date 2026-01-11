import numpy as np
import matplotlib.pyplot as plt
from ex_1 import time, time_series_y

# 3.
def MA(x, q):
    N = len(x)
    m = np.mean(x)
    errors = np.zeros(N)
    params = []
    targets = []
    
    for i in range(q, N):
        win_mean = np.mean(x[i-q:i])
        errors[i] = x[i] - win_mean
    
    for t in range(q, N):
        lagged_errors = errors[t-q:t][::-1] 
        params.append(lagged_errors)
        targets.append(x[t] - m) 
        
    theta, _, _, _ = np.linalg.lstsq(np.array(params), np.array(targets), rcond=None)
    
    predictions = np.zeros(N)
    predictions[:q] = x[:q] 
    
    for t in range(q, N):
        lag_errors = errors[t-q:t][::-1]
        predictions[t] = m + np.dot(theta, lag_errors)
        
    return predictions

q = 50

predictions = MA(time_series_y, q)
mse = np.mean((time_series_y[q:] - predictions[q:])**2)

colors = ['darkorange', 'lightseagreen']
titles = [f'Original Time Series', f'MA - Moving Average (MSE = {mse:.2f})']
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
plt.savefig('./img/Ex_3.pdf', format='pdf')
