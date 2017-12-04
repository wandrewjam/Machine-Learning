# Homework 2, Part 1

import numpy as np
import matplotlib.pyplot as plt
from important_functions import *


def flipping_experiment(num_flips=10, num_coins=10**3, num_expts=10**5):
    trials = np.random.binomial(num_flips, 0.5, size=(num_expts, num_coins))

    nu_1 = np.mean(trials[:, 0])
    nu_m = np.mean(np.amin(trials, axis=1))
    nu_r = np.mean(trials[range(num_expts), np.random.randint(num_coins, size=num_expts)])

    return nu_1, nu_m, nu_r


def regression_trial(num_points=100, low=-1, high=1, mc_points=10**3, plotting=False):
    w_true = generate_target(low=low, high=high)
    X, y = generate_sample(num_points=num_points, w_true=w_true, low=low, high=high)
    g = np.linalg.lstsq(X, y)[0]
    e_in = np.mean(np.sign(np.dot(X, g)) != y)

    e_out = monte_carlo(w=w_true, g=g, iters=mc_points)

    if plotting:
        x = np.linspace(start=low, stop=high, num=2)
        x1, x2 = X[:, 1], X[:, 2]
        plt.plot(x1[y == 1], x2[y == 1], '.')
        plt.plot(x1[y == -1], x2[y == -1], '.')
        plt.plot(x, -w_true[1]*x/w_true[2] - w_true[0]/w_true[2])
        plt.plot(x, -g[1]*x/g[2] - g[0]/g[2], 'k--')
        plt.axis([low, high, low, high])
        plt.legend(['Positive', 'Negative', 'Target', 'Hypothesis'], loc='upper center',
                   bbox_to_anchor=(0.5, 1.05),  ncol=2, shadow=True, fancybox=True)
        plt.show()

    return e_in, e_out


def regression_experiment(num_trials=10**3, num_points=100, low=-1, high=1, mc_points=10**3):
    e_in, e_out = np.empty(shape=num_trials), np.empty(shape=num_trials)
    for i in range(num_trials):
        e_in[i], e_out[i] = regression_trial(num_points=num_points, low=low, high=high,
                                             mc_points=mc_points, plotting=False)
    return np.mean(e_in), np.mean(e_out)


def pla_with_regression(num_trials=10**3, num_points=10, low=-1, high=1):
    iters = np.zeros(shape=num_trials)
    for i in range(num_trials):
        w_true = generate_target(low=low, high=high)
        X, y = generate_sample(num_points=num_points, w_true=w_true, low=low, high=high)
        g = np.linalg.lstsq(X, y)[0]

        iters[i] = pla(X=X, y=y, g_init=g)[-1]

    return np.mean(iters)


# Problems 1 & 2
nu_1, nu_m, nu_r = flipping_experiment()
print(nu_1/10, nu_m/10, nu_r/10)

# Problems 5 & 6
e_in_avg, e_out_avg = regression_experiment()
print(e_in_avg, e_out_avg)

# Problem 7
iter_mean = pla_with_regression()
print(iter_mean)
