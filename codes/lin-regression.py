# Homework 2, Part 2

import matplotlib.pyplot as plt
import numpy as np

#np.random.seed(100000)
NPOINTS = 10
MCITER = 10000
trials = 10000


def generate_target():
    # Generate the target function f
    div_points = np.random.rand(2, 2)
    w1 = (div_points[1, 1] - div_points[0, 1])/(div_points[1, 0] - div_points[0, 0])
    w0 = div_points[0, 1] - div_points[0, 0]*w1
    w = np.array([w0, w1, -1])
    return w


def generate_sample(num_points, w):
    # Generate the test points and classify
    test_points = np.zeros([num_points, 6])
    test_points[:, 0] = 1
    test_points[:, 1:3] = np.random.rand(num_points, 2)
    test_points[:, 3] = np.sign(np.dot(test_points[:, 0:3], w))
    return test_points


def pla(test_points, g_init=np.zeros(3)):
    # Implement PLA, count the iterations to convergence, and the error
    g = g_init
    iters = 0
    test_points[:, 4] = np.sign(np.dot(test_points[:, 0:3], g))
    test_points[:, 5] = test_points[:, 3] == test_points[:, 4]
    wrong = test_points[test_points[:, 5] < 1, :]
    nrows = wrong.shape[0]
    while nrows > 0:
        if nrows == 1:
            g = g + wrong[0, 0:3]*wrong[0, 3]
        else:
            rind = np.random.randint(0, nrows)
            g = g + wrong[rind, 0:3]*wrong[rind, 3]
        iters += 1
        test_points[:, 4] = np.sign(np.dot(test_points[:, 0:3], g))
        test_points[:, 5] = test_points[:, 3] == test_points[:, 4]
        wrong = test_points[test_points[:, 5] < 1, :]
        nrows = wrong.shape[0]
    return g, iters


def lin_reg(test_points):
    g = np.linalg.lstsq(test_points[:, 0:3], test_points[:, 3])[0]
    iters = 0
    test_points[:, 4] = np.sign(np.dot(test_points[:, 0:3], g))
    test_points[:, 5] = test_points[:, 3] == test_points[:, 4]
    wrong = test_points[test_points[:, 5] < 1, :]
    nrows = wrong.shape[0]

    e_in = 1 - np.mean(test_points[:, 5])

    mcpoints = np.ones([mciter, 4])
    mcpoints[:,1:3] = np.random.rand(mciter, 2)
    mcpoints[:,3] = (np.abs(np.sign(np.dot(mcpoints[:, 0:3], w))
                   - np.sign(np.dot(mcpoints[:, 0:3], g))))/2
    e_out = np.mean(mcpoints[:, 3])/2

    # # Split the test points into the two classification groups
    # positives = test_points[test_points[:,3]>0,:]
    # negatives = test_points[test_points[:,3]<0,:]
    #
    # # Plot f, g, and \mathbf{x}
    # x = np.linspace(0,1)
    # xpos = positives[:,1]
    # ypos = positives[:,2]
    # xneg = negatives[:,1]
    # yneg = negatives[:,2]
    #
    # plt.plot(xneg,yneg,'.g')
    # plt.plot(xpos,ypos,'.r')
    # plt.plot(x,w1*x+w0)
    # plt.plot(x,-g[1]/g[2]*x-g[0]/g[2],'--k')
    # plt.axis([0,1,0,1])
    # #plt.legend(['Negative','Positive'])
    # plt.show()
    return [e_in, e_out, iters]

output = np.zeros([trials, 3])
for i in range(0,trials):
    output[i, :] = lin_reg(NPOINTS, MCITER)

e_in = np.mean(output[:,0])
e_out = np.mean(output[:,1])
iters = np.mean(output[:,2])
print(e_in)
print(e_out)
print(iters)
