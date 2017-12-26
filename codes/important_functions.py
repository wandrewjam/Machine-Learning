#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:08:32 2017

@author: andrewwork
"""

# A collection of important Python functions for the Machine Learning Homework assignments

import matplotlib.pyplot as plt
import numpy as np


def generate_2D_target(low=-1, high=1, fn_type='clf'):
    # Generate the target function f
    a1, a2, b1, b2 = np.random.uniform(low=low, high=high, size=4)
    w1, w2 = a2 - b2, b1 - a1
    w0 = -w2*a2 - w1*a1
    w = np.array([w1, w2])
    if fn_type == 'clf':
        def target(x):
            return np.sign(np.dot(x, w) + w0)
    elif fn_type == 'reg':
        def target(x):
            return np.dot(x, w) + w0
    else:
        print('Please choose clf or reg for function type')
        target = None
    return target, np.r_[w0, w]


def generate_sample(num_points, target_fun, d=2, low=-1, high=1):
    # Generate the sample points and classify
    X = np.random.uniform(low=low, high=high, size=(num_points, d))
    y = target_fun(X)
    return X, y


def randomize_target(y, nu=0.1):
    num_points = y.shape[0]
    rand_indices = np.random.choice(num_points, int(num_points*nu), replace=False)
    y[rand_indices] = -y[rand_indices]
    return y


def pla(X, y, g_init=None, max_iters=10**6, pocket=False, tol=1e-6):
    # Implement PLA, count the iterations to convergence, and the error
    n_points = X.shape[1] + 1
    if g_init is None:
        g_init = np.zeros(n_points)

    g = g_init
    iters = 0
    y_guess = np.sign(np.dot(X, g[1:])) + g[0]
    wrong_loc = np.where(y != y_guess)[0]
    e_in = len(wrong_loc)/n_points

    if pocket:
        best_g = np.copy(g)
        best_err = np.copy(e_in)

    while len(wrong_loc) > 0 and max_iters > iters and e_in > tol:
        random_index = np.random.choice(wrong_loc)
        g = g + np.r_[y[random_index], X[random_index]*y[random_index]]
        iters += 1
        y_guess = np.sign(np.dot(X, g[1:]) + g[0])
        wrong_loc = np.where(y != y_guess)[0]
        e_in = len(wrong_loc)/n_points
        if pocket and e_in < best_err:
            best_err = np.copy(e_in)
            best_g = np.copy(g)

    if pocket:
        g = np.copy(best_g)

    def guess(x):
        return np.sign(np.dot(x, g[1:]) + g[0])

    return guess, e_in, iters, g


def estimate_error(target, guess, error_function='clf-det', nu=0, low=-1, high=1, iters=10**6, d=2):
    mcpoints = np.random.uniform(low=low, high=high, size=(iters, d))
    if error_function == 'clf-det':
        error = np.mean(target(mcpoints) != guess(mcpoints))
    elif error_function == 'clf-rnd':
        y = randomize_target(y=target(mcpoints), nu=nu)
        error = np.mean(y != guess(mcpoints))
    elif error_function == 'lstsq':
        error = np.mean((target(mcpoints) - guess(mcpoints))**2)
    else:
        print('Please enter a valid error function')
        error = None
    return error
