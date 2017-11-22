import numpy as np


def generate_target():
    # Generate the target function f
    div_points = np.random.uniform(low=-1, high=1, size=(2, 2))
    w1 = (div_points[1, 1] - div_points[0, 1])/(div_points[1, 0] - div_points[0, 0])
    w0 = div_points[0, 1] - div_points[0, 0]*w1
    w = np.array([w0, w1, -1])
    return w


def generate_sample(num_points, w):
    # Generate the test points and classify
    test_points = np.ones([num_points, 4])
    test_points[:, 1:3] = np.random.uniform(low=-1, high=1, size=(num_points, 2))
    test_points[:, 3] = np.sign(np.dot(test_points[:, 0:3], w))
    return test_points


def log_regression(training, g_init=np.zeros(3), tol=1e-6, max_iter=10**4, eta=0.1):
    d_E = lambda W, x_n, y_n: (-y_n * x_n)/(1+np.exp(y_n * W.dot(x_n)))

    g = np.copy(g_init)
    w = np.zeros(3)
    old_w = np.copy(w)
    iters = 0
    n_points, n_cols = training.shape
    y = training[:, -1]
    X = training[:, 0:n_cols-1]
    error = tol+1

    while True:
        old = np.copy(g)
        for i in np.random.permutation(n_points):
            xn = X[i]
            yn = y[i]
            g += eta*(yn*xn)/(1 + np.exp(yn*np.dot(xn, g)))
        iters += 1
        error = np.linalg.norm(g - old)
        error1 = np.linalg.norm(old_w - w)
        if error < tol:
            break
        old_w = np.copy(w)

    return g, iters


NTRIALS = 10
NPOINTS = 100
MCPOINTS = 10**6
mcdata = np.ones([MCPOINTS, 3])
mcdata[:, 1:3] = np.random.random([MCPOINTS, 2])
e_out = np.zeros(NTRIALS)
iters = np.zeros(NTRIALS)
for i in range(0, NTRIALS):
    w = generate_target()
    training = generate_sample(NPOINTS, w)
    g, iters[i] = log_regression(training, tol=0.01)
    g = -g/g[2]
    e_out[i] = np.mean(np.log(1+np.exp(-np.sign(np.dot(mcdata,w))*np.dot(mcdata, g))))

print(np.mean(e_out))
print(np.mean(iters))

# perfect_error = np.mean(np.log(1+np.exp(-np.sign(np.dot(mcdata,np.array([0.5,0.5,-1])))*np.dot(mcdata,np.array([0.5,0.5,-1])))))
# print(perfect_error)
