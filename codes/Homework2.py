# Homework 2, Part 1

import matplotlib.pyplot as plt
import time
import numpy as np
# np.random.seed(1000)

start = time.clock()
num_flips = 10
num_coins = 10**3
num_expts = 10**5
trials = np.random.binomial(num_flips, 0.5, size=(num_expts, num_coins))

nu_1 = np.mean(trials[:, 0])
nu_m = np.mean(np.amin(trials, axis=1))
nu_r = np.mean(trials[range(num_expts), np.random.randint(num_coins, size=num_expts)])

print(nu_1/10)
print(nu_m/10)
print(nu_r/10)
end = time.clock()

print(end-start)
