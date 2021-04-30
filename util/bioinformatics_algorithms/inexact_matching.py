import numpy as np


def smith_waterman_edit_distance(pattern, text):
    # in order to provide the start indices reverse the strings
    text = text[::-1]
    pattern = pattern[::-1]

    s = np.zeros((len(text) + 1, len(pattern) + 1))

    for j in range(1, len(pattern) + 1):
        s[0, j] = j

    for i in range(1, len(text) + 1):
        for j in range(1, len(pattern) + 1):
            s[i, j] = min(s[i, j - 1], s[i - 1, j], s[i - 1, j - 1]) + 1
            if text[i - 1] == pattern[j - 1]:
                s[i, j] = min(s[i, j], s[i - 1, j - 1])

    min_cost = np.min(s[:, -1])
    start_indices = []

    for i in range(len(text)):
        if s[len(pattern) - i, -1] == min_cost:
            start_indices.append(i)

    return start_indices
