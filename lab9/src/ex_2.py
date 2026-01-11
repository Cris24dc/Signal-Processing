import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from ex_1 import time, time_series_y

# 2.
def exponential_smoothing(x, params):
    alpha = params[0]
    s = [x[0]]

    for t in range(1, len(x)):
        s.append(alpha*x[t]+(1-alpha)*s[t-1])

    return s


def double_exponential_smoothing(x, params):
    alpha, beta = params
    s = [x[0]]
    b = [x[1] - x[0]]

    for t in range(1, len(x)):
        s.append(alpha*x[t]+(1-alpha)*(s[t-1]+b[t-1]))
        b.append(beta*(s[t]-s[t-1])+(1-beta)*b[t-1])

    return s, b


def triple_exponential_smoothing(x, params):
    alpha, beta, gamma, L = params
    
    s = np.zeros(len(x))
    b = np.zeros(len(x))
    c = np.zeros(len(x))
    result = np.zeros(len(x))

    s[L-1] = np.mean(x[:L])
    b[L-1] = (x[L-1] - x[0]) / L
    c[:L] = x[:L] - s[L-1]
    result[:L] = x[:L] 

    for t in range(L, len(x)):
        s[t] = alpha*(x[t]-c[t-L])+(1-alpha)*(s[t-1]+b[t-1])
        b[t] = beta*(s[t]-s[t-1])+(1-beta)*b[t-1]
        c[t] = gamma*(x[t]-s[t])+(1-gamma)*c[t-L]
        
        result[t] = s[t] + b[t] + c[t]
        
    return result


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def grid_search(x, funct, params):
    best_params = None
    best_mse = float('inf')
    
    for param in product(*params):
        new_series = funct(x, param)
        mse = compute_mse(x, new_series)

        if mse < best_mse:
            best_mse = mse
            best_params = param

    return best_params, best_mse


betas = np.linspace(0.01, 0.99, 100)
alphas = np.linspace(0.01, 0.99, 100)
gammas = np.linspace(0.1, 0.9, 5)
L = 50

best_alpha, mse_ses = grid_search(
    time_series_y, 
    exponential_smoothing, 
    [alphas]
)

best_params_des, mse_des = grid_search(
    time_series_y, 
    double_exponential_smoothing, 
    [alphas, betas]
)

best_params_tes, mse_tes = grid_search(
    time_series_y, 
    triple_exponential_smoothing, 
    [alphas, betas, gammas, [L]]
)

print(f"Simple Exponential Smoothing: MSE: {mse_ses:.4f}, Alpha: {best_alpha[0]:.2f}")
print(f"Double Exponential Smoothing: MSE: {mse_des:.4f}, Alpha: {best_params_des[0]:.2f}, Beta: {best_params_des[1]:.2f}")
print(f"Triple Exponential Smoothing: MSE: {mse_tes:.4f}, Alpha: {best_params_tes[0]:.2f}, Beta: {best_params_tes[1]:.2f}, Gamma: {best_params_tes[2]:.2f}, L: {best_params_tes[3]}")

y_ses = exponential_smoothing(time_series_y, best_alpha)
y_des, _ = double_exponential_smoothing(time_series_y, best_params_des)
y_tes = triple_exponential_smoothing(time_series_y, best_params_tes)

colors = ['darkorange', 'lightseagreen', 'mediumpurple', 'olivedrab']
titles = ['Original Time Series', 'Exponential Smoothing', 'Double Exponential Smoothing', 'Triple Exponential Smoothing']
Ox = [time] * 4
Oy = [time_series_y, y_ses, y_des, y_tes]

fig, axs = plt.subplots(4, 1, figsize=(12, 12))

for i in range(0, 4):
    axs[i].plot(Ox[i], Oy[i], colors[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Amplitude')
    axs[i].grid()

plt.tight_layout()
plt.savefig('./img/Ex_2.pdf', format='pdf')