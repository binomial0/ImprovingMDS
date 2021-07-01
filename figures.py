from time import time
from collections import OrderedDict

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances

from mod_algorithms import _minimize_lbfgsb, _smacof_single

# Code to generate the figures

# This is less mature than the implementations in main.py, and some internals of the libraries were used in order to
# be able to plot the stress values. Slightly modified versions of some algorithms are in in mod_algorithms.py.

dataset_name = "covtype"
n_points = 2000  # if the dataset is larger, only n_points random points are used
normalize_columns = True
rand = 43
show_stress_it = True
show_stress_time = True
n_components = 2

it_counts = np.array([20, 50, 200])
#it_counts = np.array([10, 30, 100])
#it_counts = np.array([5, 20, 50])

# Uncomment the methods that should be used
def get_methods():
    ret = OrderedDict()
    for i in it_counts:
        ret[f"smacof-{i}it"] = scikit_mds_alt(i), "smacof"
    #for i in it_counts:
    #    ret[f"bfgs-nochunk-{i}it"] = l_bfgs_b(i, chunked=False), "lbfgs_nc"
    for i in it_counts:
        ret[f"lbfgs-{i}it"] = l_bfgs_b(i, chunked=True), "lbfgs"
    #for i in it_counts:
    #    ret[f"bfgs-chunk64-{i}it"] = l_bfgs_b(i, chunked=True, chunk_size=64), "lbfgs64"
    #for i in it_counts:
    #    ret[f"bfgs-chunk128-{i}it"] = l_bfgs_b(i, chunked=True, chunk_size=128), "lbfgs128"
    #for i in it_counts:
    #    ret[f"lbfgs-chunk256-{i}it"] = l_bfgs_b(i, chunked=True, chunk_size=256), "lbfgs256"
    #for i in it_counts:
    #    ret[f"bfgs-chunk512-{i}it"] = l_bfgs_b(i, chunked=True, chunk_size=512), "lbfgs512"
    #for i in it_counts:
    #    ret[f"bfgs-chunkl-{i}it"] = l_bfgs_b(i, chunked=True, precompute_dissim=False), "lbfgs_l"
    for i in it_counts:
        ret[f"sgd-{i}it"] = sgd_epochs(i, chunk_size=256), "sgd"
    # for i in it_counts:
    #    ret[f"sgd-{i}it"] = sgd_epochs(i, chunk_size=64), "sgd"
    #for i in it_counts:
    #    ret[f"sgd-128-{i}it"] = sgd_epochs(i, chunk_size=128), "sgd128"
    #for i in it_counts:
    #    ret[f"sgd-256-{i}it"] = sgd_epochs(i, chunk_size=256), "sgd256"
    #for i in it_counts:
    #    ret[f"sgd-512-{i}it"] = sgd_epochs(i, chunk_size=512), "sgd512"
    # here go more methods
    return ret


# Show diagrams from the Results section
def main():
    data, target = load_dataset(dataset_name)
    show(data, target, dataset_name)


# Show the s-curve visualisation from the introduction
def main_v():
    data, target = load_dataset('s_curve')
    fig = plt.figure(figsize=(8, 4))
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.scatter(data[:, 0], data[:, 1], data[:, 2], c=target, cmap=plt.cm.Spectral)
    ax3d.view_init(4, -72)
    ax3d.xaxis.set_major_formatter(mticker.NullFormatter())
    ax3d.yaxis.set_major_formatter(mticker.NullFormatter())
    ax3d.zaxis.set_major_formatter(mticker.NullFormatter())

    method = l_bfgs_b(200, chunked=True)
    Y, _, _ = method(data)
    ax = fig.add_subplot(122)
    ax.xaxis.set_major_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.NullFormatter())
    ax.scatter(Y[:, 0], Y[:, 1], c=target, cmap=plt.cm.Spectral)

    plt.show()


def load_dataset(name):
    if name == "s_curve":
        # S curve (3-dimensional)
        data, target = datasets.make_s_curve(n_points, random_state=rand)
    elif name == "swiss_roll":
        # swiss roll (3-dimensional)
        data, target = datasets.make_swiss_roll(n_points, random_state=rand)
    elif name == "blobs":
        # 10 blobs in 5-dimensional space
        data, target = datasets.make_blobs(n_points, n_features=5, centers=10, random_state=rand)
    else:
        # loaded datasets
        if name == "iris":
            # 150 samples with 4 dimensions in 3 classes
            # good results
            data, target = datasets.load_iris(return_X_y=True)
        elif name == "digits":
            # 1797 samples with 64 dimensions in 10 classes
            # ok results, not completely clear
            data, target = datasets.load_digits(return_X_y=True)
        elif name == "wine":
            # 178 samples with 13 dimensions in 3 classes
            # good results
            data, target = datasets.load_wine(return_X_y=True)
        elif name == "cancer":
            # 569 samples with 30 dimensions in 2 classes
            # quite good results
            data, target = datasets.load_breast_cancer(return_X_y=True)
        elif name == "newsgroups":
            # 18846 samples with 130107 dimensions in 20 classes
            # no structured results at all up to 2000 samples (23s)
            data, target = datasets.fetch_20newsgroups_vectorized(return_X_y=True, normalize=False)
            data = data.toarray()
        elif name == "covtype":
            # 581012 samples with 54 dimensions in 7 classes
            # weird results up to 2000 samples (doesn't look meaningful, but there's some structure)
            data, target = datasets.fetch_covtype(return_X_y=True)
        else:
            raise Exception("invalid dataset name")
        if len(data) > n_points:
            perm = random.permutation(len(data))
            perm = perm[:n_points]
            data = data[perm]
            target = target[perm]
    if normalize_columns:
        # We usually work with heterogenous data, so the columns are given in different units.
        # This leads to arbitrary differences in magnitude in the data columns.
        # Often, one column has a range from 0 to 1000, while the others range from 0 to 1,
        #  so our data looks like one large worm with little discernible features, because only one column is important.
        # To remedy this, we normalize all columns to range from 0 to 1.
        #  (To be fair, "every column is equally important" is still an arbitrary decision,
        #  but more reasonable than before.)
        min_values = np.amin(data, axis=0, keepdims=True)
        max_values = np.amax(data, axis=0, keepdims=True)
        amplitudes = np.maximum(max_values - min_values, 0.01)  # make sure we don't divide by zero
        data = (data - min_values) / amplitudes

    return data, target


def show(X, color, name):
    methods = get_methods()
    num_methods = len(methods)
    rows = (num_methods // 3)+show_stress_it+show_stress_time
    figure = plt.figure(figsize=(9, 3*rows))
    minimums = np.full(3, np.inf)
    maximums = np.zeros(3)
    nullform = mticker.NullFormatter()
    iterform = mticker.FormatStrFormatter('%dit')
    secondform = mticker.FormatStrFormatter('%.1fs')
    if show_stress_it:
        stress_plots = []
        for i in range(3):
            stress_plots.append(figure.add_subplot(rows, 3, num_methods + i + 1))
    if show_stress_time:
        time_plots = []
        for i in range(3):
            time_plots.append(figure.add_subplot(rows, 3, num_methods+i+1+3*show_stress_it))
    for i, (label, (method, shortname)) in enumerate(methods.items()):
        t0 = time()
        Y, stress, times = method(X)
        t1 = time()
        print(f"{label}: {(t1-t0):.2g} sec")
        if isinstance(stress, list):
            stress = compute_stress(X, stress)
        axis = figure.add_subplot(rows, 3, i+1)
        axis.set_title(f"{label} ({(t1-t0):.2g} s)")
        axis.scatter(Y[:, 0], Y[:, 1], c=color, s=10)
        axis.xaxis.set_major_formatter(nullform)
        axis.yaxis.set_major_formatter(nullform)
        if show_stress_it:
            stressplot = stress_plots[i % 3]
            stressplot.set_title(f"{np.size(stress)}it stress")
            stressplot.xaxis.set_major_formatter(iterform)
            stressplot.plot(stress, label=shortname)
            if i % 3 == 2:
                stressplot.legend()

        if show_stress_time:
            stplot = time_plots[i%3]
            stplot.set_title(f"stress vs time")
            stplot.xaxis.set_major_formatter(secondform)
            stplot.plot(times-t0, stress, label=shortname)
            if i % 3 == 2:
                stplot.legend()
            min = stress[-1]
            max = stress[np.size(stress)//10]
            if min < minimums[i % 3]:
                minimums[i % 3] = min
                stplot.set_ylim(min * 0.95, maximums[i % 3])
                if show_stress_it:
                    stressplot.set_ylim(min * 0.95, maximums[i % 3])
            if max > maximums[i % 3]:
               maximums[i % 3] = max
               stplot.set_ylim(minimums[i % 3] * 0.95, max)
               if show_stress_it:
                   stressplot.set_ylim(minimums[i % 3] * 0.95, max)
    figure.tight_layout(h_pad=1)


    plt.savefig(f"output/{name}.pdf")
    plt.show()


def scikit_mds_alt(max_iter):
    save_stress = np.zeros(max_iter)
    save_time = np.zeros(max_iter)
    def inner(X):
        dissim = euclidean_distances(X)
        res, stress, n_iter = _smacof_single(dissim, save_stress=save_stress, save_time=save_time, max_iter=max_iter, random_state=rand, eps=-1)
        print("Stress: ", 2*stress)
        print("iter: ", n_iter)
        return res, 2*save_stress, save_time
    return inner


def sgd_epochs(max_iter, chunk_size=256):
    rng = np.random.RandomState(rand)
    save_time = np.zeros(max_iter)
    iterations = []

    def inner(X):
        n = np.size(X, axis=0)
        x_norm_squared = (X ** 2).sum(axis=1)
        Y = np.random.RandomState(rand).rand(n, 2)
        for iter in range(max_iter):
            stepsize = (1-0.5*np.sqrt(iter/max_iter))/n
            permutation = rng.permutation(n)
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
            save_time[iter] = time()

            iterations.append(Y.copy())
        return Y, iterations, save_time
    return inner

def compute_stress(X, iterations):
    stress = np.zeros(len(iterations))
    target_distances = euclidean_distances(X)
    for (i, Y) in enumerate(iterations):
        distances = euclidean_distances(Y)
        offsets = target_distances - distances
        sq_offsets = offsets * offsets
        stress[i] = np.sum(sq_offsets)
    return stress


def l_bfgs_b(max_iter, cg = False, chunked=True, chunk_size=256, precompute_dissim=True):
    def inner(X):
        if precompute_dissim:
            target_distances = euclidean_distances(X)
        else:
            target_normsq = (X**2).sum(axis=1)
        init = np.random.RandomState(rand).rand(np.size(X, axis=0)*2)

        save_stress = np.zeros(max_iter)
        save_time = np.zeros(max_iter)

        def weight(Y):
            Y = np.reshape(Y, (-1, 2))
            distances = euclidean_distances(Y)
            offsets = target_distances - distances
            sq_offsets = offsets*offsets
            value = np.sum(sq_offsets)

            np.fill_diagonal(distances, np.inf)
            factor = -4*offsets / distances

            grad = Y * factor.sum(axis=1)[:, None] - np.matmul(factor, Y)
            grad = np.ravel(grad)
            return value, grad

        def weight_chunked(Y):
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

        if chunked:
            result = _minimize_lbfgsb(weight_chunked, init, jac=True, maxiter=max_iter, ftol=0, gtol=0, save_stress=save_stress, save_time=save_time)
            print("Stress: ", result.fun)
            print("iter: ", result.nit)
        else:
            result = _minimize_lbfgsb(weight, init, jac=True, maxiter=max_iter, ftol=0, gtol=0, save_stress=save_stress,
                                      save_time=save_time)
            print("Stress: ", result.fun)
            print("iter: ", result.nit)
        return np.reshape(result.x, (-1, 2)), save_stress, save_time
    return inner


main()
#main_v()