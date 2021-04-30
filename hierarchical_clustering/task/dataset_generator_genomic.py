import argparse
import os
import pickle
import numpy as np

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import string_to_list


class HierarchicalClusteringGenomicDatasetGenerator:

    def __init__(self, strings):
        length = max(len(s) for s in strings)
        sequences = [string_to_list(s, length=length) for s in strings]

        distances = cross_distance_matrix_threads(strings, strings, 5) / length
        similarities = 1 - distances

        # plt.hist(x=np.min(distances, axis=0), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        # plt.show()

        self.sequences_leaves = np.asarray(sequences)
        self.similarities = similarities

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.sequences_leaves, self.similarities), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="./data/hc_qiita_large_extr.pkl", help='Output data path')
    parser.add_argument('--N_leaves', type=int, default=10000, help='Number of sequences')
    parser.add_argument('--source_sequences', type=str, default='./data/qiita.txt', help='Sequences path')
    args = parser.parse_args()

    with open(args.source_sequences, 'rb') as f:
        L = f.readlines()
    L = [l[:-1].decode('UTF-8') for l in L]
    # L = L[7700:] # for extrapolation

    strings = L[:(args.N_leaves if args.N_leaves > 0 else len(L))]

    data = HierarchicalClusteringGenomicDatasetGenerator(strings=strings)
    data.save_as_pickle(args.out)
