# Homework 2, Part 1

import numpy as np
import matplotlib.pyplot as plt
from important_functions import *


def flipping_experiment(num_flips=10, num_coins=10**3, num_expts=10**5):
    trials = np.random.binomial(num_flips, 0.5, size=(num_expts, num_coins))

    nu_1 = np.mean(trials[:, 0])
    nu_m = np.mean(np.amin(trials, axis=1))
    nu_r = np.mean(trials[range(num_expts), np.random.randint(num_coins, size=num_expts)])

    print('The average value of nu_1 is %.2f, the average value of nu_min is %.2f, and the average value of nu_rand is '
          '%.2f' % (nu_1/num_flips, nu_m/num_flips, nu_r/num_flips))

    return None


def regression_trial(num_points=100, low=-1, high=1, mc_points=10**3, plotting=False):
    target, w_true = generate_2D_target(low=low, high=high)
    X, y = generate_sample(num_points=num_points, target_fun=target, d=2, low=low, high=high)
    g = np.linalg.lstsq(np.c_[np.ones(shape=X.shape[0]), X], y)[0]

    def guess(x):
        return np.sign(np.dot(x, g[1:]) + g[0])
    y_guess = guess(X)
    e_in = np.mean(y_guess != y)

    e_out = estimate_error(target=target, guess=guess, iters=mc_points)

    if plotting:
        x = np.linspace(start=low, stop=high, num=2)
        x1, x2 = X[:, 0], X[:, 1]
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
    print('For the linear regression classification algorithm on 100 points, the average E_in was %.4f and the average '
          'E_out was %.4f' % (np.mean(e_in), np.mean(e_out)))

    return None


def pla_with_regression(num_trials=10**3, num_points=10, low=-1, high=1):
    iters = np.zeros(shape=num_trials)
    for i in range(num_trials):
        target = generate_2D_target(low=low, high=high)[0]
        X, y = generate_sample(num_points=num_points, target_fun=target, d=2, low=low, high=high)
        g = np.linalg.lstsq(np.c_[np.ones(shape=X.shape[0]), X], y)[0]

        iters[i] = pla(X=X, y=y, g_init=g)[-2]

    print('The PLA for 10 training points converged in %.2f iterations on average when using the least-squares '
          'regression solution as the initial guess' % (np.mean(iters)))
    return None


def target_fn(x):
    return np.sign(x[:, 0]**2 + x[:, 1]**2 - 0.6)


def nl_trfm_trial(target_fn, num_points=10**3, nu=0.1, low=-1, high=1, plotting=False):
    def quad_transform(x):
        z = np.ones(shape=(x.shape[0], 5))
        z[:, :2] = x
        z[:, 2] = x[:, 0]*x[:, 1]
        z[:, 3:] = x**2
        return z

    X, y = generate_sample(num_points=num_points, target_fun=target_fn, d=2, low=low, high=high)
    Z = quad_transform(X)
    y = randomize_target(y, nu=0.1)
    g_lin = np.linalg.lstsq(np.c_[np.ones(shape=X.shape[0]), X], y)[0]
    g_quad = np.linalg.lstsq(np.c_[np.ones(shape=Z.shape[0]), Z], y)[0]

    def guess_lin(x):
        return np.sign(np.dot(x, g_lin[1:]) + g_lin[0])

    def guess_quad(x):
        return np.sign(np.dot(quad_transform(x), g_quad[1:]) + g_quad[0])

    e_in_lin = np.mean(guess_lin(X) != y)
    e_in_quad = np.mean(guess_quad(X) != y)
    e_out_lin = estimate_error(target=target_fn, guess=guess_lin, error_function='clf-rnd', nu=nu,
                               d=2, low=low, high=high)
    e_out_quad = estimate_error(target=target_fn, guess=guess_quad, error_function='clf-rnd', nu=nu,
                                d=2, low=low, high=high)

    if plotting:
        x = np.linspace(start=low, stop=high, num=2)
        x1, x2 = X[:, 0], X[:, 1]
        plt.plot(x1[y == 1], x2[y == 1], '.')
        plt.plot(x1[y == -1], x2[y == -1], '.')
        plt.plot(x, -g_lin[1]*x/g_lin[2] - g_lin[0]/g_lin[2], 'k--')
        plt.axis([low, high, low, high])
        plt.legend(['Positive', 'Negative', 'Target', 'Hypothesis'], loc='upper center',
                   bbox_to_anchor=(0.5, 1.05),  ncol=2, shadow=True, fancybox=True)
        plt.show()


    return e_in_lin, e_in_quad, e_out_lin, e_out_quad, -g_quad/g_quad[0]


def nl_trfm_problem(target_fn, num_points=10**3, num_trials=10**3, low=-1, high=1):
    e_in_lin, e_in_quad, e_out_lin, e_out_quad = np.zeros(shape=num_trials), np.zeros(shape=num_trials), \
                                                 np.zeros(shape=num_trials), np.zeros(shape=num_trials)
    g_quad_mat = np.zeros(shape=(num_trials, 6))

    for i in range(num_trials):
        e_in_lin[i], e_in_quad[i], e_out_lin[i], e_out_quad[i], g_quad_mat[i] = \
            nl_trfm_trial(target_fn=target_fn, num_points=num_points, low=low, high=high, plotting=False)

    g_avg = np.mean(g_quad_mat, axis=0)

    def quad_transform(x):
        z = np.ones(shape=(x.shape[0], 5))
        z[:, :2] = x
        z[:, 2] = x[:, 0]*x[:, 1]
        z[:, 3:] = x**2
        return z

    def guess_quad(x):
        return np.sign(np.dot(quad_transform(x), g_avg[1:]) + g_avg[0])

    avg_err = estimate_error(target=target_fn, guess=guess_quad, error_function='clf-rnd', nu=0.1, low=low, high=high)

    print('For linear regression without transformation, the average E_in was %.4f. The average hypothesis for linear '
          'regression with transformation was \ng = %s. The E_out of this hypothesis was %.4f.'
          % (np.mean(e_in_lin), g_avg, avg_err))

    return None


# Problems 1 & 2
flipping_experiment()
print('')

# Problems 5 & 6
regression_experiment(num_trials=10**4)
print('')

# Problem 7
pla_with_regression(num_trials=10**4)
print('')

# Problems 8, 9, & 10
nl_trfm_problem(target_fn=target_fn)
