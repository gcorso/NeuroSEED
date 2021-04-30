import argparse
import os
import pickle
import random
import time
from tqdm import tqdm
from multiprocessing import Pool
import torch
import numpy as np
import matplotlib.pyplot as plt

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix
from util.data_handling.string_generator import IndependentGenerator, k_mutations


def generate_batch(args):
    # generates a single batch of sequences and computes their distance matrix
    batch_size, len_sequence, string_generator, max_changes = args
    sequences = [string_generator.generate(len_sequence)]
    for i in range(batch_size - 1):
        S_ref = sequences[random.randint(0, i)]
        S = k_mutations(S_ref, 1 + np.random.geometric(max_changes / len_sequence))
        sequences.append(S)

    sequences_str = ["".join(chr(s + 97) for s in S) for S in sequences]
    distances = cross_distance_matrix(sequences_str, sequences_str)
    return sequences, distances


class EditDistanceDatasetGenerator:

    def __init__(self, N_batches, batch_size, len_sequence, max_changes, string_generator, n_thread=5, seed=0, plot=False):
        self.sequences = {}
        self.distances = {}
        random.seed(seed)

        for dataset in N_batches.keys():
            print("Generating", dataset, end=':')

            # parallel batch generation
            with Pool(n_thread) as pool:
                start = time.time()
                batches = list(
                    tqdm(
                        pool.imap(generate_batch, [(batch_size[dataset], len_sequence[dataset], string_generator,
                                                    max_changes[dataset]) for _ in range(N_batches[dataset])]),
                        total=N_batches[dataset],
                        desc="Batches " + dataset,
                    ))
                print("Time to compute the batches: {}".format(time.time() - start))

            # concatenate all batches
            batches = list(zip(*batches))
            sequence_batches = batches[0]
            distance_batches = batches[1]

            # plot histogram of distances
            if plot:
                plt.hist(x=np.reshape(np.asarray(distance_batches), (-1)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
                plt.show()

            self.sequences[dataset] = torch.from_numpy(np.asarray(sequence_batches)).long()
            self.distances[dataset] = torch.from_numpy(np.asarray(distance_batches)).float()

    def save_as_pickle(self, filename):
        directory = os.path.dirname(filename)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.sequences, self.distances), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="../data/edit_synthetic_small.pkl", help='Output data path')
    parser.add_argument('--train_size', type=int, default=1400, help='Number of training batches')
    parser.add_argument('--val_size', type=int, default=200, help='Number of validation batches')
    parser.add_argument('--test_size', type=int, default=400, help='Number of test batches')
    parser.add_argument('--batch_size', type=int, default=50, help='Sequences per batch')
    parser.add_argument('--len_sequence', type=int, default=1024, help='Length of the sequences')
    parser.add_argument('--max_changes', type=float, default=13, help='Parameter of changes distribution')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    generator = IndependentGenerator(seed=args.seed)
    data = EditDistanceDatasetGenerator\
        (N_batches={"train": args.train_size, "val": args.val_size, "test": args.test_size},
         batch_size={"train": args.batch_size, "val": args.batch_size, "test": args.batch_size},
         len_sequence={"train": args.len_sequence, "val": args.len_sequence, "test": args.len_sequence},
         max_changes={"train": args.max_changes, "val": args.max_changes, "test": args.max_changes},
         seed=args.seed, string_generator=generator)
    data.save_as_pickle(args.out)

