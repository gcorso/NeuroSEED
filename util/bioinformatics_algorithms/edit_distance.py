import time
from functools import partial

import Levenshtein
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def dp_edit_distance(s1, s2):
    # Not used because too slow, instead use Levenshtein.distance from a library written in C
    s = np.zeros((len(s1) + 1, len(s2) + 1))

    for i in range(1, len(s1) + 1):
        s[i, 0] = i
    for j in range(1, len(s2) + 1):
        s[0, j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            s[i, j] = min(s[i, j - 1], s[i - 1, j], s[i - 1, j - 1]) + 1
            if s1[i - 1] == s2[j - 1]:
                s[i, j] = min(s[i, j], s[i - 1, j - 1])
    return s[len(s1), len(s2)]


def cross_distance_row(args, distance=Levenshtein.distance):
    a, B = args
    return [distance(a, b) for b in B]


def cross_distance_matrix_threads(A, B, n_thread, distance=Levenshtein.distance):
    with Pool(n_thread) as pool:
        start = time.time()
        distance_matrix = list(
            tqdm(
                pool.imap(partial(cross_distance_row, distance=distance), zip(A, [B for _ in A])),
                total=len(A),
                desc="Edit distance {}x{}".format(len(A), len(B)),
            ))
        print("Time to compute the matrix: {}".format(time.time() - start))
        return np.array(distance_matrix)


def cross_distance_matrix(A, B, distance=Levenshtein.distance):
    return np.array(np.array([[distance(a, b) for b in B] for a in A]))
