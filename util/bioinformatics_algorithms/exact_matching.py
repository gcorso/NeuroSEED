def brute_force(pattern, text):
    matches = []
    for i in range(len(text) - len(pattern) + 1):
        different = False
        for j in range(len(pattern)):
            if pattern[j] != text[i + j]:
                different = True
                break
        if not different:
            matches.append(i)
    return matches


def z_preprocessing(S):
    """
    Z algorithm for the fundamental preprocessing of a string:
    Given a string S and a position i > 1, let Z_i(S) be the length of the longest
    substring of S that starts at i and matches a prefix of S.
    """
    n = len(S)
    z = [0] * n
    x, y = 0, 0
    for i in range(1, n):
        z[i] = max(0, min(z[i - x], y - i + 1))
        while i + z[i] < n and S[z[i]] == S[i + z[i]]:
            x = i
            y = i + z[i]
            z[i] += 1
    return z


def z_matching(pattern, text):
    assert not ('$' in pattern or '$' in text), '$ is the special character reserved for separation'

    combined = pattern + ['$'] + text
    z = z_preprocessing(combined)
    matches = [idx - len(pattern) - 1 for idx in range(len(z)) if z[idx] == len(pattern)]
    return matches


def boyer_moore_preprocessing(pattern, alphabet_size=4):
    """
    Bad character rule used by Boyer-Moore algorithm:
    For each character x in the alphabet, let R(x) be the position of right-most occurrence of character x in P.
    R(x) is defined to be zero if x does not occur in P.
    """
    R = [0] * alphabet_size
    for i in range(len(pattern)):
        R[pattern[i]] = i
    return R


def boyer_moore_matching(pattern, text):
    R = boyer_moore_preprocessing(pattern)
    matches = []
    k = 0
    while k < len(text) - len(pattern) + 1:
        match = True
        for i in range(len(pattern)-1, -1, -1):
            if text[k+i] != pattern[i]:
                k += max(1, i - R[text[k+i]])
                match = False
                break
        if match:
            matches.append(k)
            k += 1

    return matches


def kmp_preprocessing(pattern):
    """
    Knuth-Morris-Pratt algorithm shift preprocessing:
    For each position i in pattern P, define T_i(P) to be the length of the longest
    proper suffix of P[l..i] that matches a prefix of P.
    """
    T = [0] * (len(pattern)+1)
    pos = 1
    cnd = 0

    T[0] = -1
    while pos < len(pattern):
        if pattern[pos] == pattern[cnd]:
            T[pos] = T[cnd]
        else:
            T[pos] = cnd
            cnd = T[cnd]
            while cnd >= 0 and pattern[pos] != pattern[cnd]:
                cnd = T[cnd]
        pos += 1
        cnd += 1
    T[pos] = cnd
    return T


def kmp_matching(pattern, text):
    T = kmp_preprocessing(pattern)

    matches = []
    j = 0
    k = 0

    while j < len(text):
        if pattern[k] == text[j]:
            j += 1
            k += 1
            if k == len(pattern):
                matches.append(j-k)
                k = T[k]
        else:
            k = T[k]
            if k < 0:
                j += 1
                k += 1
    return matches