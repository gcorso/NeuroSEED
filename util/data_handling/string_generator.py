import random

DNA_ALPHABET = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

PROTEIN_ALPHABET = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
    'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}


def string_to_list(s, length=None, alphabet=DNA_ALPHABET):
    L = [alphabet[s[i]] if i<len(s) else -1
         for i in range(length if length is not None else len(s))]
    return L


class IndependentGenerator:
    def __init__(self, alphabet_size=4, seed=None):
        if seed is not None: random.seed(seed)
        self.alphabet_size = alphabet_size

    def generate(self, length):
        string = [random.randint(0, self.alphabet_size - 1) for _ in range(length)]
        return string


def k_mutations(S, k, alphabet_size=4):
    S1 = S.copy()  # shallow-cloning
    while k > 0:
        change = random.randint(0, 1)

        if change == 0:  # substitution
            idx = random.randint(0, len(S1) - 1)
            S1[idx] = random.randint(0, alphabet_size - 1)
            k -= 1

        elif change == 1:  # remove and insert
            idx = random.randint(0, len(S1) - 1)
            S1.pop(idx)
            idx = random.randint(0, len(S1))
            S1.insert(idx, random.randint(0, alphabet_size - 1))
            k -= 2

    return S1
