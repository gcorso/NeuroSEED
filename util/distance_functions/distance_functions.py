import torch
import torch.nn as nn
import numpy as np


def square_distance(t1_emb, t2_emb):
    D = t1_emb - t2_emb
    d = torch.sum(D * D, dim=-1)
    return d


def euclidean_distance(t1_emb, t2_emb):
    D = t1_emb - t2_emb
    d = torch.norm(D, dim=-1)
    return d


def cosine_distance(t1_emb, t2_emb):
    return 1 - nn.functional.cosine_similarity(t1_emb, t2_emb, dim=-1, eps=1e-6)


def manhattan_distance(t1_emb, t2_emb):
    D = t1_emb - t2_emb
    d = torch.sum(torch.abs(D), dim=-1)
    return d


def hyperbolic_distance(u, v, epsilon=1e-7):  # changed from epsilon=1e-7 to reduce error
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)


def hyperbolic_distance_numpy(u, v, epsilon=1e-9):
    sqdist = np.sum((u - v) ** 2, axis=-1)
    squnorm = np.sum(u ** 2, axis=-1)
    sqvnorm = np.sum(v ** 2, axis=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = np.sqrt(x ** 2 - 1)
    return np.log(x + z)


DISTANCE_TORCH = {
    'square': square_distance,
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'manhattan': manhattan_distance,
    'hyperbolic': hyperbolic_distance
}
