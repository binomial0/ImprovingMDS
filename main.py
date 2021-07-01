
# The MDS implementations
# The functions smacof, lbfgs and sgd return the respecitve solvers.
# Usage:
#   method = smacof(max_iterations, 42) # or lbfgs or sgd
#   embedding = method(X)
# X is a ndarray with dimensions (n, d)
# the returned embedding is a ndarray with dimensions (n, 2)

import numpy as np
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from scipy.optimize import minimize


def smacof(max_iter, random_state=None, eps=0):
    method = manifold.MDS(n_components=2, max_iter=max_iter, n_init=1, random_state=random_state, metric=True, eps=eps)
    def inner(X):
        return method.fit_transform(X)
    return inner

def lbfgs(max_iter, random_state=None, eps=0, chunk_size=256, precompute_dissim=True):
    def inner(X):
        n = np.size(X, axis=0)
        if precompute_dissim:
            target_distances = euclidean_distances(X)
        else:
            target_normsq = (X**2).sum(axis=1)
        init = np.random.RandomState(random_state).rand(2*n)

        def fg(Y):
            Y = np.reshape(Y, (-1, 2))
            n = np.size(Y, axis=0)
            normdiff = np.zeros((n, 2))
            value = 0
            norm_squared = (Y**2).sum(axis=1)
            for i in range(0, n, chunk_size):
                chunkA = Y[i:(i+chunk_size)]
                for j in range(0, i+1, chunk_size):
                    chunkB = Y[j:(j+chunk_size)]
                    distances = euclidean_distances(chunkA, chunkB,
                                                    X_norm_squared=norm_squared[i:(i+chunk_size), None],
                                                    Y_norm_squared=norm_squared[None, j:(j+chunk_size)])
                    if precompute_dissim:
                        target_chunk = target_distances[i:(i+chunk_size),j:(j+chunk_size)]
                    else:
                        target_chunk = euclidean_distances(X[i:(i+chunk_size)], X[j:(j+chunk_size)],
                                                           X_norm_squared=target_normsq[i:(i+chunk_size), None],
                                                           Y_norm_squared=target_normsq[None, j:(j+chunk_size)],)
                    offsets = target_chunk - distances
                    sq_offsets = offsets * offsets
                    v = np.sum(sq_offsets)
                    value += v

                    if i == j:
                        np.fill_diagonal(distances, np.inf)
                    factor = -4 * offsets / distances
                    normdiff[i:(i+chunk_size)] += chunkA * factor.sum(axis=1)[:, None] - np.matmul(factor, chunkB)
                    if i != j:
                        value += v
                        normdiff[j:(j + chunk_size)] += chunkB * factor.sum(axis=0)[:, None] - np.matmul(factor.transpose(), chunkA)
            grad = np.ravel(normdiff)
            return value, grad

        result = minimize(fg, init, jac=True, method='L-BFGS-B', tol=eps, options={'maxiter': max_iter})
        return np.reshape(result.x, (n, 2))
    return inner

def sgd(max_iter, random_state=None, chunk_size=256):
    rng = np.random.RandomState(random_state)
    def inner(X):
        n = np.size(X, axis=0)
        x_norm_squared = (X ** 2).sum(axis=1)
        Y = rng.rand(n, 2)
        for iter in range(max_iter):
            stepsize = (1-0.5*np.sqrt(iter/max_iter))/n
            permutation = rng.permutation(n)
            value = 0
            Xperm = X[permutation]
            Xnsperm = x_norm_squared[permutation]
            Yperm = Y[permutation]
            for i in range(0, n, chunk_size):
                for j in range(0, i+1, chunk_size):
                    chunkA = Yperm[i:(i+chunk_size)]
                    chunkB = Yperm[j:(j+chunk_size)]
                    distances = euclidean_distances(chunkA, chunkB)
                    target_chunk = euclidean_distances(Xperm[i:(i + chunk_size)], Xperm[j:(j + chunk_size)],
                                                       X_norm_squared=Xnsperm[i:(i + chunk_size), None],
                                                       Y_norm_squared=Xnsperm[None, j:(j + chunk_size)],)
                    offsets = target_chunk - distances

                    if i == j:
                        np.fill_diagonal(distances, np.inf)
                    factor = -4 * offsets / distances
                    Yperm[i:(i + chunk_size)] -= stepsize * (chunkA * factor.sum(axis=1)[:, None] - np.matmul(factor, chunkB))
                    if i != j:
                        Yperm[j:(j + chunk_size)] -= stepsize * (chunkB * factor.sum(axis=0)[:, None] - np.matmul(factor.transpose(), chunkA))
            Y[permutation] = Yperm
        return Y
    return inner


