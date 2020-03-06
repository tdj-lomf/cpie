"""CPIE helper functions
"""

import numpy as np

def mean_and_covariance(solutions):
    """Calculate mean vector and covariance matrix of the solutions
    Arguments:
        solutions {list of Solution} -- target
    Returns:
        (numpy array, numpy matrix) -- Mean vector and covariance matrix
    """
    sol_size = len(solutions)
    dimension = solutions[0].x.size
    # calc mean
    mu = sum(s.x for s in solutions) / len(solutions)
    #calc variance
    variance = np.zeros((dimension, dimension))
    for s in solutions:
        diff = s.x - mu
        variance += np.outer(diff, diff)
    variance /= sol_size
    return mu, variance

# ToDo Speed up with numpy
def k_means_mahalanobis(solutions, k=2, max_iter=100):
    """k-means clustering with mahalanobis distance.
    Arguments:
        solutions {list of Solution} -- Clustering target
    Keyword Arguments:
        k {int} -- Number of division (default: {2})
        max_iter {int} -- Max inner iteration (default: {100})
    Returns:
        list of (list of Solution) -- k clustered solutions
    """
    assert k <= len(solutions), "k should be less than or equarl to len(solutions)"
    seed_bases = np.random.choice(solutions, size=k, replace=False)
    seeds = [s.x.copy() for s in seed_bases]
    dimension = seeds[0].size
    BInvs = [np.eye(dimension) for i in range(k)]
    class_indices = [0] * len(solutions)
    for t in range(max_iter):
        is_changed = False
        classes = [[] for i in range(k)]
        # classify solutions
        for i, s in enumerate(solutions):
            distances = [np.linalg.norm(np.dot(BInv, s.x - seed)) 
                         for seed, BInv in zip(seeds, BInvs)]
            class_index = np.argmin(distances)
            if class_index != class_indices[i]:
                is_changed = True
            class_indices[i] = class_index
            classes[class_index].append(s)
        if not is_changed or t == max_iter - 1:
            break
        # update
        for i, c in enumerate(classes):
            if len(c) == 0:
                seeds[i] = np.zeros(dimension)
                BInvs[i] = np.eye(dimension)
            elif len(c) <= dimension:
                seeds[i] = sum(s.x for s in c) / len(c)
                BInvs[i] = np.eye(dimension)
            else:
                seeds[i], covariance = mean_and_covariance(c)
                BInvs[i] = np.linalg.inv(np.linalg.cholesky(covariance))
    return classes
