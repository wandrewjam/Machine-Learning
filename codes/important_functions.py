#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:08:32 2017

@author: andrewwork
"""

# A collection of important Python functions for the Machine Learning Homework assignments

import matplotlib.pyplot as plt
import numpy as np


def generate_target(low=0, high=1):
    # Generate the target function f
    a1, a2, b1, b2 = np.random.uniform(low=low, high=high, size=4)
    w1, w2 = a2 - b2, b1 - a1
    w0 = -w2*a2 - w1*a1
    w_true = np.array([w0, w1, w2])
    return w_true


def generate_sample(num_points, w_true, low=0, high=1):
    # Generate the sample points and classify
    X = np.ones(shape=(num_points, w_true.shape[0]))
    X[:, 1:] = np.random.uniform(low=low, high=high, size=(num_points, w_true.shape[0] - 1))
    y = np.sign(np.dot(X, w_true))
    return X, y


def pla(X, y, g_init=None, max_iters=10**6, pocket=False, tol=1e-6):
    # Implement PLA, count the iterations to convergence, and the error
    n_points = X.shape[1]
    if g_init is None:
        g_init = np.zeros(n_points)

    g = g_init
    iters = 0
    y_guess = np.sign(np.dot(X, g))
    wrong_loc = np.where(y != y_guess)[0]
    e_in = len(wrong_loc)/n_points

    if pocket:
        best_g = np.copy(g)
        best_err = np.copy(e_in)

    while len(wrong_loc) > 0 and max_iters > iters and e_in > tol:
        random_index = np.random.choice(wrong_loc)
        g = g + X[random_index]*y[random_index]
        iters += 1
        y_guess = np.sign(np.dot(X, g))
        wrong_loc = np.where(y != y_guess)[0]
        e_in = len(wrong_loc)/n_points
        if pocket and e_in < best_err:
            best_err = np.copy(e_in)
            best_g = np.copy(g)

    if pocket:
        g = np.copy(best_g)

    return g, iters


def monte_carlo(w, g, low=0, high=1, iters=10**6):
    mcpoints = np.ones([iters, len(w)])
    mcpoints[:, 1:] = np.random.uniform(low=low, high=high, size=(iters, len(w) - 1))
    error_vector = np.sign(np.dot(mcpoints, w)) != np.sign(np.dot(mcpoints, g))
    error = np.mean(error_vector)
    return error
