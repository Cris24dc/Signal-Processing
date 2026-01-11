import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from ex_1 import time, time_series_y

def ARMA(x, p, q):
    N = len(x)
    m = np.mean(x)
    errors = np.zeros(N)
    win_size = max(p, q, 40)
    centered_x = x - m
    params = []
    targets = []
    start_ind = max(p, q)

    for i in range(win_size, N):
        errors[i] = x[i] - np.mean(x[i-win_size:i])

    for t in range(start_ind, N):
        ar_lags = centered_x[t-p:t][::-1]
        ma_lags = errors[t-q:t][::-1]
        
        params.append(np.concatenate([ar_lags, ma_lags]))
        targets.append(centered_x[t])

    coeffs, _, _, _ = np.linalg.lstsq(np.array(params), np.array(targets), rcond=None)
    
    predictions = np.zeros(N)
    predictions[:start_ind] = x[:start_ind]

    for t in range(start_ind, N):
        ar_lags = (predictions[t-p:t] - m)[::-1] 
        ma_lags = errors[t-q:t][::-1]
        
        lags = np.concatenate([ar_lags, ma_lags])
        predictions[t] = m + np.dot(coeffs, lags)

    mse = np.mean((x[start_ind:] - predictions[start_ind:])**2)
    aic = N * np.log(mse) + 2 * (p + q + 1)

    return predictions, mse, aic


p_values = range(1, 21)
q_values = range(1, 21)

best_aic = float('inf')
best_params = (0, 0)
for p, q in product(p_values, q_values):
    _, _, aic = ARMA(time_series_y, p, q)
    
    if aic < best_aic:
        best_aic = aic
        best_params = (p, q)

best_p, best_q = best_params


# ARMA without trend
final_predictions, final_mse, _ = ARMA(time_series_y, best_p, best_q)
# ARIMA
model = ARIMA(time_series_y, order=(best_p, 1, best_q))
results = model.fit()
final_predictions_ARIMA = results.predict(start=0, end=len(time_series_y)-1, typ='levels')


colors = ['darkolivegreen', 'darkorange', 'rebeccapurple']
titles = ['Original Time Series', f'ARMA (p:{best_p}, q:{best_q}, MSE:{final_mse:.2f})', 'ARIMA']
Ox = [time] * 3
Oy = [time_series_y, final_predictions, final_predictions_ARIMA]

fig, axs = plt.subplots(3, 1, figsize=(12, 10))

for i in range(3):
    axs[i].plot(Ox[i], Oy[i], color=colors[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Amplitude')
    axs[i].grid()

plt.tight_layout()
plt.savefig('./img/Ex_4.pdf', format='pdf')
