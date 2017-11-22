import numpy as np
np.random.seed(1000)


def target_line():
    ''' generate two points, construct two equations , solve for W of the target line'''
    w0 = 1
    point_1 , point_2 = np.random.uniform(-1., 1., size=(2,2))
    # A*W = b
    A = [point_1 , point_2]
    b= [-w0,-w0]
    w1, w2 = np.linalg.solve(A, b)

    return np.array([w0, w1, w2])


def generate_data(N):
    ''' generate data , x0 = 1, x1 & x2 random '''
    data = np.random.uniform(-1., 1., size=(N,2))
    data_set =  np.column_stack((np.ones(N), data))

    return data_set


def prediction(data, W):
    '''classify points '''
    labels = np.sign(data.dot(W))
    return labels


def logistic_regression(D, y_labels):

    d_E = lambda W, x_n, y_n: (-y_n * x_n)/(1+np.exp(y_n * W.dot(x_n)))

    W = np.array([0., 0., 0.])
    old_W = np.copy(W)
    eta = 0.01
    epocs = 0

    while True:
        for i in np.random.permutation(len(y_labels)):
            x_n = D[i]
            y_n = y_labels[i]
            W -= eta*d_E(W,x_n,y_n)

        epocs += 1
        if np.linalg.norm(W - old_W) < 0.01:
            break

        old_W = np.copy(W)

    return W,epocs


# Testing The Algorithm
def cross_entropy(D_test,y_labels,W):
    e = 0
    N = len(y_labels)
    for i in range(N):
        y_n = y_labels[i]
        x_n = D_test[i]

        e += np.log(1+np.exp(-y_n * W.dot(x_n)))

    E_out = e / N
    return E_out

def logistic_test(n):
    E = []
    epocs = []
    for i in range(100):
        target = target_line()
        D = generate_data(100)
        y_labels = prediction(D,target)
        W,epoc = logistic_regression(D,y_labels)

        D_test = generate_data(n)
        test_labels = prediction(D_test,target)


        E_out = cross_entropy(D_test,test_labels,W)

        E.append(E_out)
        epocs.append(epoc)

    average_error = np.mean(E)
    average_epocs = np.mean(epocs)
    return average_error,average_epocs


average_error,average_epocs = logistic_test(100)
print('Error:', average_error)
print('Epochs: ', average_epocs)
