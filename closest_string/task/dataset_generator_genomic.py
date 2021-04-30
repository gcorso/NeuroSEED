import argparse
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import string_to_list


class ClosestStringGenomicDatasetGenerator:

    def __init__(self, strings_reference, strings_query, n_queries, plot=False):
        # compute maximum length and transform sequences into list of integers
        length = max(len(s) for s in strings_reference + strings_query)
        sequences_references = [string_to_list(s, length=length) for s in strings_reference]
        sequences_queries = [string_to_list(s, length=length) for s in strings_query]

        # compute distances and find reference with minimum distance
        distances = cross_distance_matrix_threads(strings_reference, strings_query, 5)
        minimum = np.min(distances, axis=0, keepdims=True)

        # queries are only valid if there is a unique answer (no exaequo)
        counts = np.sum((minimum+0.5 > distances).astype(float), axis=0)
        valid = counts == 1
        labels = np.argmin(distances, axis=0)[valid][:n_queries]

        # print an histogram of the minimum distances
        if plot:
            plt.hist(x=np.min(distances, axis=0)[valid], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.show()

        # convert to torch
        self.sequences_references = torch.from_numpy(np.asarray(sequences_references)).long()
        self.sequences_queries = torch.from_numpy(np.asarray(sequences_queries)[valid][:n_queries]).long()
        self.labels = torch.from_numpy(labels).float()

        print("Shapes:", "References", self.sequences_references.shape, " Queries", self.sequences_queries.shape,
              " Labels", self.labels.shape)

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.sequences_references, self.sequences_queries, self.labels), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="./data/closest_qiita_large.pkl", help='Output data path')
    parser.add_argument('--N_reference', type=int, default=1000, help='Number of reference sequences')
    parser.add_argument('--test_query', type=int, default=2000, help='Query sequences tested (some may be discarded)')
    parser.add_argument('--N_queries', type=int, default=2000, help='Number of queries')
    parser.add_argument('--source_sequences', type=str, default='./data/qiita.txt', help='Sequences data path')
    args = parser.parse_args()

    # load sequences
    with open(args.source_sequences, 'rb') as f:
        L = f.readlines()
    L = [l[:-1].decode('UTF-8') for l in L]
    # L = L[7700:] add for extrapolation from edit distance training

    strings_reference = L[:args.N_reference]
    strings_queries = L[args.N_reference: (len(L) if args.test_query < 0 else args.N_reference + args.test_query)]

    data = ClosestStringGenomicDatasetGenerator(strings_reference, strings_queries, args.N_queries)
    data.save_as_pickle(args.out)

