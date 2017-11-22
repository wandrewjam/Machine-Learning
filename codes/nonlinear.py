# Homework 2, Part 3

import matplotlib.pyplot as plt
import numpy as np

#np.random.seed(100000)
NPOINTS = 1000
MCITER = 100000
trials = 1000


def nonlinear(npoints, mciter):
    test_points = np.zeros([npoints, 9])
    test_points[:, 0] = 1
    test_points[:, 1:3] = np.random.rand(npoints, 2)
    test_points[:, 3] = test_points[:, 1]*test_points[:, 2]
    test_points[:, 4] = test_points[:, 1]**2
    test_points[:, 5] = test_points[:, 2]**2
    test_points[:, 6] = np.sign(test_points[:, 1]**2 + test_points[:, 2]**2 - 0.6)

    rand_inds = np.random.randint(npoints, size=npoints//10)
    test_points[rand_inds, 6] = -test_points[rand_inds, 6]

    g = np.linalg.lstsq(test_points[:, 0:6], test_points[:, 6])[0]
    g = -g/g[0]
    # iters = 0
    test_points[:, 7] = np.sign(np.dot(test_points[:, 0:6], g))
    test_points[:, 8] = test_points[:, 3] != test_points[:, 4]

    e_in = np.mean(test_points[:, 8])
    mcpoints = np.ones([mciter, 7])
    mcpoints[:, 1:3] = np.random.rand(mciter, 2)
    mcpoints[:, 3] = mcpoints[:, 1]*mcpoints[:, 2]
    mcpoints[:, 4] = mcpoints[:, 1]**2
    mcpoints[:, 5] = mcpoints[:, 2]**2
    gest = np.sign(np.dot(mcpoints[:, 0:6], g))
    mcpoints[:, 6] = np.abs(np.sign(mcpoints[:, 1]**2 + mcpoints[:, 2]**2 - 0.6) - gest)
    err_1 = np.mean(np.abs(np.sign(np.dot(mcpoints[:, 0:6], np.array([-1, -0.05, 0.08, 0.13, 1.50, 1.50]))) - gest))/2
    err_2 = np.mean(np.abs(np.sign(np.dot(mcpoints[:, 0:6], np.array([-1, -0.05, 0.08, 0.13, 1.50, 15.0]))) - gest))/2
    err_3 = np.mean(np.abs(np.sign(np.dot(mcpoints[:, 0:6], np.array([-1, -0.05, 0.08, 0.13, 15.0, 1.50]))) - gest))/2
    err_4 = np.mean(np.abs(np.sign(np.dot(mcpoints[:, 0:6], np.array([-1, -1.50, 0.08, 0.13, 0.05, 0.05]))) - gest))/2
    err_5 = np.mean(np.abs(np.sign(np.dot(mcpoints[:, 0:6], np.array([-1, -0.05, 0.08, 1.50, 0.15, 0.15]))) - gest))/2
    e_out = np.mean(mcpoints[:, 6])/2

    # # Split the test points into the two classification groups
    # positives = test_points[test_points[:,6]>0,:]
    # negatives = test_points[test_points[:,6]<0,:]
    #
    # # Plot f, g, and \mathbf{x}
    # x = np.linspace(0, 1)
    # X1, X2 = np.meshgrid(x, x)
    # xpos = positives[:, 1]
    # ypos = positives[:, 2]
    # xneg = negatives[:, 1]
    # yneg = negatives[:, 2]
    #
    # plt.plot(xneg, yneg, '.g')
    # plt.plot(xpos, ypos, '.r')
    # # plt.plot(x,w1*x+w0)
    # # plt.plot(x, -g[1]/g[2]*x - g[0]/g[2], '--k')
    # plt.plot(np.sqrt(0.6)*np.cos(np.pi*x/2), np.sqrt(0.6)*np.sin(np.pi*x/2))
    # plt.contour(X1, X2, g[0] + g[1]*X1 + g[2]*X2 + g[3]*X1*X2 + g[4]*X1**2 + g[5]*X2**2, [0])
    # plt.axis([0, 1, 0, 1])
    # #plt.legend(['Negative','Positive'])
    # plt.show()
    return e_out, err_1, err_2, err_3, err_4, err_5

output = np.zeros([trials, 6])
for i in range(0,trials):
    output[i, :] = nonlinear(NPOINTS, MCITER)

means = np.mean(output, axis=0)
print(means)

# g = nonlinear(NPOINTS, MCITER)
# print(g)
