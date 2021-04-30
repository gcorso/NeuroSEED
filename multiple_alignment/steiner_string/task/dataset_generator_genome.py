import argparse
import os
import pickle
import torch
import numpy as np

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import string_to_list


class MSAPairDatasetGeneratorGenome:

    def __init__(self, strings, length):
        self.datasets = {}

        for dataset in ['train', 'val']:
            print("Generating", dataset)

            sequences = [string_to_list(s, length=length) for s in strings[dataset]]
            distances = cross_distance_matrix_threads(strings[dataset], strings[dataset], 5)

            self.datasets[dataset] = {}
            self.datasets[dataset]['texts'] = torch.from_numpy(np.asarray(sequences)).long()
            self.datasets[dataset]['distances'] = torch.from_numpy(distances).float()

        for dataset in ['val_msa', 'test']:
            sequences = []
            print("Generating", dataset)

            for S in strings[dataset]:
                sequences_batch = [string_to_list(s, length=length) for s in S]
                sequences.append(sequences_batch)

            self.datasets[dataset] = {}
            self.datasets[dataset]['texts'] = torch.from_numpy(np.asarray(sequences)).long()

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump(self.datasets, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="../data/msa_qiita_small.pkl", help='Data path.')
    parser.add_argument('--train_size', type=int, default=100, help='')
    parser.add_argument('--val_size', type=int, default=10, help='')
    parser.add_argument('--val_msa_size', type=int, default=5, help='')
    parser.add_argument('--test_size', type=int, default=10, help='')
    parser.add_argument('--number_sequences', type=int, default=30, help='')
    parser.add_argument('--source_sequences', type=str, default='../../data/qiita.txt', help='Data path.')
    args = parser.parse_args()

    with open(args.source_sequences, 'rb') as f:
        L = f.readlines()
    L = [l[:-1].decode('UTF-8') for l in L]
    #L = list(set(L))

    assert args.train_size + args.val_size + args.test_size * args.number_sequences <= len(L)

    strings = {
        'train': L[:args.train_size],
        'val': L[args.train_size:args.train_size + args.val_size],
        'val_msa': [L[args.train_size + args.val_size + i * args.number_sequences:
                   args.train_size + args.val_size + (i + 1) * args.number_sequences]
                 for i in range(args.val_msa_size)],
        'test': [L[args.train_size + args.val_size + (args.val_msa_size + i) * args.number_sequences:
                   args.train_size + args.val_size + (args.val_msa_size + i + 1) * args.number_sequences]
                 for i in range(args.test_size)]
    }

    length = max(len(s) for s in L)

    data = MSAPairDatasetGeneratorGenome(strings=strings, length=length)
    data.save_as_pickle(args.out)
