from random import sample, randint
from scipy.spatial.distance import euclidean
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math

colours = ['r', 'b' ,'g', 'y']

def kmeans(k, samples, centroids=[], distance_func=euclidean):
    """ Apply the Kmeans algorithm on samples with k centroids. """
    assert(len(centroids) <= k)
    assert(len(samples) > 0)
    centroids = get_centroids(k, samples, centroids)

    converged = False

    while not converged:
        total_sum_dist = 0
        clusters = defaultdict(list)

        for s in samples:
            cluster_idx = np.argmin([distance_func(s, c) for c in centroids])
            clusters[cluster_idx].append(s)

        old_centroids = centroids.copy()
        
        for idx in range(len(centroids)):
            x, y = list(zip(*clusters[idx]))
            plt.plot(x, y, colours[idx] + 'o')
            centroids[idx] = tuple([round(n, 2) for n in np.mean(clusters[idx], axis=(0))])
            plt.plot(*centroids[idx], colours[idx] + '^')

        plt.show(block=False)
        plt.pause(5)
        plt.close()

        converged = all([centroid in old_centroids for centroid in centroids])

    print(centroids)
    return centroids

def get_centroids(k, samples, base_centroids=[]):
    centroids = base_centroids.copy()
    samples_not_in_centroids = [s for s in samples if s not in base_centroids]
    centroids.extend(sample(samples_not_in_centroids, k-len(base_centroids)))
    return centroids

X = [(randint(0,10), randint(0,10)) for _ in range(30)]
#X = [(1,1), (1,2), (2,1), (8,1), (8,2), (7,1), (1,8),(2,8),(1,7)]
kmeans(3, X)