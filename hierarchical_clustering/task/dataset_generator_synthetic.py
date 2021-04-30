import argparse
import os
import pickle
import random

import numpy as np

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import IndependentGenerator, k_mutations


class HierarchicalClusteringDatasetGenerator:

    def __init__(self, N_reference, N_leaves, len_sequence, min_changes, max_changes, string_generator, seed=0):
        random.seed(seed)

        # generate root sequence
        sequences_references = [string_generator.generate(len_sequence)]
        sequences_leaves = []

        # generates a set of reference (internal nodes of the tree) which will then be discarded
        for i in range(N_reference - 1):
            S_ref = sequences_references[random.randint(0, len(sequences_references)-1)]
            S = k_mutations(S_ref, np.random.randint(min_changes, max_changes))
            sequences_references.append(S)

        # generate leaves sequences starting from the references
        for i in range(N_leaves):
            S_ref = sequences_references[random.randint(0, N_reference-1)]
            S = k_mutations(S_ref, np.random.randint(min_changes, max_changes))
            sequences_leaves.append(S)

        # compute distances and convert to similarities
        sequences_leaves_str = ["".join(chr(s + 97) for s in S) for S in sequences_leaves]
        distances = cross_distance_matrix_threads(sequences_leaves_str, sequences_leaves_str, 5) / len_sequence
        similarities = 1 - distances

        #plt.hist(x=np.min(distances, axis=0), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        #plt.show()

        self.sequences_leaves = np.asarray(sequences_leaves)
        self.similarities = similarities

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.sequences_leaves, self.similarities), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="./data/hc_1024_large.pkl", help='Output data path')
    parser.add_argument('--N_reference', type=int, default=400, help='Number of tree internal sequences')
    parser.add_argument('--N_leaves', type=int, default=2000, help='Number of sequences for hc')
    parser.add_argument('--len_sequence', type=int, default=1024, help='Length of each sequence')
    parser.add_argument('--min_changes', type=float, default=25, help='Min number of changes')
    parser.add_argument('--max_changes', type=float, default=150, help='Max number of changes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    generator = IndependentGenerator(seed=args.seed)
    data = HierarchicalClusteringDatasetGenerator\
        (N_reference=args.N_reference, N_leaves=args.N_leaves,
         len_sequence=args.len_sequence, min_changes=args.min_changes, max_changes=args.max_changes,
         seed=args.seed, string_generator=generator)
    data.save_as_pickle(args.out)

