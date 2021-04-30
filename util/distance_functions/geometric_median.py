from functools import partial

from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from util.distance_functions.distance_functions import hyperbolic_distance_numpy

distance_convertion = {
    'euclidean': 'euclidean',
    'square': 'sqeuclidean',
    'cosine': 'cosine',
    'manhattan': 'cityblock',
    'hyperbolic': hyperbolic_distance_numpy
}


def mean_distances(center, points, distance='euclidean'):
    return cdist([center], points, metric=distance_convertion[distance]).mean()


def geometric_median(points, distance='euclidean'):
    centroid = points.mean(axis=0)  # start from centroid

    if distance == 'hyperbolic':
        #constraint = NonlinearConstraint(fun=np.norm, lb=0, ub=1-1e-6)
        optimize_result = minimize(partial(mean_distances, points=points, distance=distance), centroid)
    else:
        optimize_result = minimize(partial(mean_distances, points=points, distance=distance), centroid, method='COBYLA')
    return optimize_result.x, optimize_result.fun
