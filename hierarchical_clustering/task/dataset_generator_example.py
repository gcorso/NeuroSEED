import argparse
import os
import pickle
import numpy as np

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import string_to_list, PROTEIN_ALPHABET


class HierarchicalClusteringGenomicDatasetGenerator:

    def __init__(self, sequences):
        length = max(len(s) for s in sequences)
        texts = [string_to_list(s, length=length, alphabet=PROTEIN_ALPHABET) for s in sequences]

        distances = cross_distance_matrix_threads(sequences, sequences, 5) / length
        similarities = 1 - distances

        self.texts_leaves = np.asarray(texts)
        self.similarities = similarities

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.texts_leaves, self.similarities), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="../data/hc_example.pkl", help='Output data path')
    parser.add_argument('--source_sequences', type=str, default='../data/animals.fasta', help='Sequences path')
    args = parser.parse_args()

    with open(args.source_sequences, 'rb') as f:
        L = f.readlines()
    L = [l[:-2].decode('UTF-8') for i, l in enumerate(L) if i%2 == 1]

    data = HierarchicalClusteringGenomicDatasetGenerator(sequences=L)
    data.save_as_pickle(args.out)
