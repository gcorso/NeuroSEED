import argparse
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import string_to_list


class EditDistanceGenomicDatasetGenerator:

    def __init__(self, strings, n_thread=5, plot=False):
        self.sequences = {}
        self.distances = {}

        # find maximum string length
        length = max(max(len(s) for s in seq) for seq in strings.values())

        for dataset in strings.keys():
            print("Generating", dataset)

            # convert strings to lists
            sequences = [string_to_list(s, length=length) for s in strings[dataset]]
            distances = cross_distance_matrix_threads(strings[dataset], strings[dataset], n_thread)

            # plot histogram of distances in dataset
            if plot:
                plt.hist(x=np.reshape(distances, (-1)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
                plt.show()

            # convert to torch
            self.sequences[dataset] = torch.from_numpy(np.asarray(sequences)).long()
            self.distances[dataset] = torch.from_numpy(distances).float()

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.sequences, self.distances), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="./data/edit_qiita_large.pkl", help='Output data path')
    parser.add_argument('--train_size', type=int, default=7000, help='Training sequences')
    parser.add_argument('--val_size', type=int, default=700, help='Validation sequences')
    parser.add_argument('--test_size', type=int, default=1500, help='Test sequences')
    parser.add_argument('--source_sequences', type=str, default='./data/qiita.txt', help='Sequences data path')
    args = parser.parse_args()

    # load and divide sequences
    with open(args.source_sequences, 'rb') as f:
        L = f.readlines()
    L = [l[:-1].decode('UTF-8') for l in L]

    strings = {
        'train': L[:args.train_size],
        'val': L[args.train_size:args.train_size + args.val_size],
        'test': L[args.train_size + args.val_size:args.train_size + args.val_size + args.test_size]
    }

    data = EditDistanceGenomicDatasetGenerator(strings=strings)
    data.save_as_pickle(args.out)

