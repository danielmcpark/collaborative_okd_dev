import errno
import os
import sys
import time
import math
import torch.nn as nn
import numpy as np

from sklearn.metrics import pairwise_distances
#from sklearn.metrics import rbf_kernel

__all__ = ['AverageMeter', 'RunningAverage', 'Clustering']

class RunningAverage(object):
    def __init__(self):
        self.steps = 0
        self.totals = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total/float(self.steps)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.agg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def aggregate(self, val, n_branches, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.agg = self.sum / (self.count * n_branches)

class Clustering(nn.Module):
    def __init__(self, args):
        self.Kernels=10

    def acc(self, ypred, y):
        assert len(y) > 0
        assert len(np.unique(ypred)) == len(np.unique(y))

        s = np.unique(ypred)
        t = np.unique(y)

        N = len(np.unique(ypred))
        F = np.zeros((N, N), dtype = np.int32)
        for i in range(N):
            for j in range(N):
                idx = np.logical_and(ypred == s[i], y == t[j])
                F[i][j] = np.count_nonzero(idx)

        return F

    # assign to the clusters (M-step)
    def get_assignments(self, X, centroids):
        dist = pairwise_distances(X, centroids)
        assign = np.argmin(dist, axis=1)
        return assign

    # compute the new centroids (E-step)
    def get_centroids(self, X, assignments):
        centroids = []
        for i in np.unique(assignments):
            centroids.append(X[assignments==i].mean(axis=0))
        return np.array(centroids)

    # initialize the centroids
    def init_kmeans_plus_plus(self, X, K):
        assert K>=2, "You want to make 1 cluster?"
        compute_distance = lambda X, c: pairwise_distances(X, c).min(axis=1)
        centroids = [X[np.random.choice(range(X.shape[0])),:]]
        for _ in range(K-1):
            proba = compute_distance(X, centroids)**2
            proba /= proba.sum()
            centroids.append(X[np.random.choice(range(X.shape[0]), p=proba)])
        return np.array(centroids)

    def KMeans(self, X, centroids, n_iterations=5, axes=None):
        if axes is not None:
            axes = axes.flatten()
        for i in range(n_iterations):
            assignments = self.get_assignments(X, centroids)
            centroids = self.get_centroids(X, assignments)
        return assignments, centroids

    def spectral_clustering(self, A, K=2, gamma=10):
        A /= A.sum(axis=1)
        #A = rbf_kernel_(X, gamma=gamma)
        #A -= np.eye(A.shape[0])
        A = np.multiply(A, (np.ones((10,10))-np.identity(10)))
        D = A.sum(axis=1)
        D_inv = np.diag(D**(-.5))
        L = (D_inv).dot(A).dot(D_inv) # laplacian
        s, Vh = np.linalg.eig(L)
        eigenvector = Vh.real[:,:K].copy()
        eigenvector /= ((eigenvector**2).sum(axis=1)[:,np.newaxis]**.5)
        centroids = self.init_kmeans_plus_plus(eigenvector, K)
        assignments, _ = self.Kmeans(eigenvector, centroids)
        return assignments
