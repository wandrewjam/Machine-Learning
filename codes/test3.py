import numpy as np
import matplotlib.pyplot as plt


def cluster(X, mu):
    cluster_indices = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        best_distance = np.inf
        best_index = mu.shape[0]
        for j in range(mu.shape[0]):
            if np.linalg.norm(X[i] - mu[j]) < best_distance:
                best_index = j
                best_distance = np.linalg.norm(X[i] - mu[j])
        cluster_indices[i] = best_index
    return cluster_indices


def lloyd(X, k, mu=None):
    # Initialize centers randomly
    if mu == None:
        x1min, x2min = np.amin(X, axis=0)
        x1max, x2max = np.amax(X, axis=0)
        mu = np.empty(shape=(k, X.shape[1]))
        mu[:, 0] = np.random.uniform(low=x1min, high=x1max, size=k)
        mu[:, 1] = np.random.uniform(low=x2min, high=x2max, size=k)

    cluster_indices = cluster(X, mu)
    print(cluster_indices)
    for j in range(k):
        if j not in cluster_indices:
            mu, cluster_indices = lloyd(X, k)
            return mu, cluster_indices

    iterate = True
    while iterate:
        for j in range(k):
            mu[j] = np.mean(X[cluster_indices == j], axis=0)

        new_cluster_indices = cluster(X, mu)
        print(new_cluster_indices)
        for j in range(k):
            if j not in cluster_indices:
                # If one of the clusters is empty, start over
                mu, cluster_indices = lloyd(X, k)
                return mu, cluster_indices

        iterate = not np.all(new_cluster_indices == cluster_indices)
        cluster_indices = np.copy(new_cluster_indices)

    return mu, cluster_indices


X = np.random.uniform(low=-1, high=1, size=(10, 2))
mu, cluster_indices = lloyd(X, 2)
print(mu, cluster_indices)
