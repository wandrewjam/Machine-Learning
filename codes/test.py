import numpy as np

def E(x):
    u = x[0]
    v = x[1]
    return (u*np.exp(v) - 2*v*np.exp(-u))**2


def grad_E(x):
    grad = np.zeros(2)
    u = x[0]
    v = x[1]
    grad[0] = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u))
    grad[1] = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
    return grad


def coord_descent(E, grad_E, init=np.ones(2), eta=0.1, tol=1e-6, max_iter=10**5):
    iters=0
    u = init
    error = E(u)
    while error > tol and iters < max_iter:
        u[0] = u[0] - eta*grad_E(u)[0]
        u[1] = u[1] - eta*grad_E(u)[1]
        error = E(u)
        iters = iters+1
        print(grad_E(u))
        print(u)
    return u, iters, error


u, iters, error = coord_descent(E, grad_E, init=np.ones(2), tol=1e-30, max_iter=15)
print(u, iters, error)
