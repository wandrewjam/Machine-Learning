import numpy as np
import matplotlib.pyplot as plt
from important_functions import *


def pla_trial(num_points=100, low=0, high=1, plotting=False):
    target, w = generate_2D_target(low=low, high=high)
    X, y = generate_sample(num_points=num_points, target_fun=target, d=2, low=low, high=high)
    guess, _, iters, g = pla(X=X, y=y)
    e_out = estimate_error(target=target, guess=guess, error_function='clf', low=low, high=high, d=2)

    if plotting:
        x = np.linspace(start=low, stop=high, num=2)
        x1, x2 = X[:, 0], X[:, 1]
        plt.plot(x1[y == 1], x2[y == 1], '.')
        plt.plot(x1[y == -1], x2[y == -1], '.')
        plt.plot(x, -w[1]*x/w[2] - w[0]/w[2])
        plt.plot(x, -g[1]*x/g[2] - g[0]/g[2], 'k--')
        plt.axis([low, high, low, high])
        plt.legend(['Positive', 'Negative', 'Target', 'Hypothesis'], loc='upper center',
                   bbox_to_anchor=(0.5, 1.05),  ncol=2, shadow=True, fancybox=True)
        plt.show()

    print('The out of sample classification error is %.4f, and the Perceptron Learning Algorithm '
          'converged in %d iterations' % (e_out, iters))
    return None


def pla_experiment(num_trials=1000, num_points=100, low=0, high=1):
    iters, e_out = np.empty(shape=num_trials), np.empty(shape=num_trials)
    for i in range(0, num_trials):
        target = generate_2D_target(low=low, high=high)[0]
        X, y = generate_sample(num_points=num_points, target_fun=target, d=2, low=low, high=high)
        guess, iters[i] = pla(X=X, y=y)[::2]
        e_out[i] = estimate_error(target=target, guess=guess, low=low, high=high)
    print('For %d trials on %d points, the average number of iterations was %.f and the average '
          'classification error was %.6f' % (num_trials, num_points, np.mean(iters), np.mean(e_out)))
    return None


pla_trial(num_points=10, low=0, high=1, plotting=True)
print('')
pla_experiment(num_points=10, low=0, high=1)
print('')
pla_trial(num_points=100, low=0, high=1, plotting=True)
print('')
pla_experiment(num_points=100, low=0, high=1)
