import argparse
import os
import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import IndependentGenerator, k_mutations


class ClosestStringDatasetGenerator:

    def __init__(self, N_reference, N_query, len_sequence, min_changes, max_changes, string_generator, initials,
                 seed=0, plot=False):
        random.seed(seed)

        # generate some sequences independently at random
        sequences_references = [string_generator.generate(len_sequence) for _ in range(initials)]
        sequences_queries = []

        # generate the rest of references from previously generated ones
        for i in range(N_reference - initials):
            S_ref = sequences_references[random.randint(0, len(sequences_references)-1)]
            S = k_mutations(S_ref, np.random.randint(min_changes, max_changes))
            sequences_references.append(S)

        # generate queries via random mutations
        for i in range(N_query):
            S_ref = sequences_references[random.randint(0, N_reference-1)]
            S = k_mutations(S_ref, np.random.randint(min_changes, max_changes))
            sequences_queries.append(S)

        # convert to strings for distance routine
        sequences_references_str = ["".join(chr(s + 97) for s in S) for S in sequences_references]
        sequences_queries_str = ["".join(chr(s + 97) for s in S) for S in sequences_queries]

        # compute distances and reference with minimum distance
        distances = cross_distance_matrix_threads(sequences_references_str, sequences_queries_str, 5)
        labels = np.argmin(distances, axis=0)

        # plot histogram of minimum distances
        if plot:
            plt.hist(x=np.min(distances, axis=0), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.show()

        # convert to torch
        self.sequences_references = torch.from_numpy(np.asarray(sequences_references)).long()
        self.sequences_queries = torch.from_numpy(np.asarray(sequences_queries)).long()
        self.labels = torch.from_numpy(labels).float()

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.sequences_references, self.sequences_queries, self.labels), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="./data/closest_large_1024.pkl", help='Output data path')
    parser.add_argument('--N_reference', type=int, default=10000, help='Number of references')
    parser.add_argument('--N_query', type=int, default=1000, help='Number of queries')
    parser.add_argument('--len_sequence', type=int, default=1024, help='Length of sequences')
    parser.add_argument('--min_changes', type=float, default=50, help='Minimum number of mutations')
    parser.add_argument('--max_changes', type=float, default=600, help='Maximum number of mutations')
    parser.add_argument('--initials', type=float, default=200, help='Initial independently generated sequences')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    generator = IndependentGenerator(seed=args.seed)
    data = ClosestStringDatasetGenerator\
        (N_reference=args.N_reference, N_query=args.N_query,
         len_sequence=args.len_sequence, min_changes=args.min_changes, max_changes=args.max_changes,
         seed=args.seed, string_generator=generator, initials=args.initials)
    data.save_as_pickle(args.out)

