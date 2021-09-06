import numpy as np
import scipy.spatial.distance
import torch
import torch.nn.functional as F

from util.distance_functions.distance_functions import hyperbolic_distance


def euclidean_matrix(enc_reference, enc_query, scaling=None):
    distances = torch.cdist(enc_reference, enc_query)
    return distances


def square_matrix(enc_reference, enc_query, scaling=None):
    d = euclidean_matrix(enc_reference, enc_query)
    return d * d


def manhattan_matrix(enc_reference, enc_query, scaling=None):
    distances = torch.cdist(enc_reference, enc_query, p=1)
    return distances


def cosine_matrix(enc_reference, enc_query, scaling=None):
    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    cosine_sim = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        cosine_sim[:, j] = F.cosine_similarity(enc_reference, enc_query[j:j + 1].repeat(N, 1))
    return 1 - cosine_sim


def hyperbolic_matrix(enc_reference, enc_query, scaling=None):
    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = hyperbolic_distance(enc_reference, enc_query[j:j+1].repeat(N, 1))

    if scaling is not None:
        d = d.detach().cpu() * scaling.detach().cpu()
    return d


def hyperbolic_matrix_numpy(u, v, eps=1e-9):
    m = scipy.spatial.distance.cdist(u, v) ** 2
    u_sqr = np.sum(u ** 2, axis=1)
    v_sqr = np.sum(v ** 2, axis=1)
    divisor = np.maximum(1. - np.expand_dims(u_sqr, axis=1), eps) * np.maximum(1. - np.expand_dims(v_sqr, axis=0), eps)
    D_ij = np.arccosh(1 + 2 * m / divisor)
    return D_ij


DISTANCE_MATRIX = {
    'euclidean': euclidean_matrix,
    'square': square_matrix,
    'manhattan': manhattan_matrix,
    'cosine': cosine_matrix,
    'hyperbolic': hyperbolic_matrix
}
