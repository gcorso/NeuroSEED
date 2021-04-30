import argparse
import pickle

import torch
from Bio.Phylo.TreeConstruction import _DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
import re
import time
import numpy as np
import sys

from util.distance_functions.distance_matrix import DISTANCE_MATRIX
from closest_string.test import embed_strings
from edit_distance.task.dataset import EditDistanceDatasetComplete
from multiple_alignment.steiner_string.models.loss import remove_padding


DNA_ALPHABET = ['A', 'C', 'G', 'T']


def torch_to_string(S, alphabet_size=4):
    S = S.tolist()
    S = [remove_padding(s, alphabet_size) for s in S]
    S = [[DNA_ALPHABET[l] for l in s] for s in S]
    S = [''.join(s) for s in S]
    return S


def save_fasta(sequences):
    strings = torch_to_string(sequences)
    names = ['S' + str(i) for i in range(len(strings))]
    with open('sequences.fasta', 'w') as f:
        for i in range(len(strings)):
            f.write('> ' + names[i] + '\n')
            f.write(strings[i] + '\n')


def remove_inner_nodes_tree(filename):
    with open(filename, 'rb') as f:
        t = f.readline()[:-1].decode('UTF-8')
    t = re.sub('Inner[0-9]+', '', t)
    with open(filename, 'w') as f:
        f.write(t)


def build_guide_trees(distance_matrix):
    # build distance matrix biopython object
    matrix = [distance_matrix[i, :i + 1].tolist() for i in range(len(distance_matrix))]
    names = ['S' + str(i) for i in range(len(distance_matrix))]
    dm = _DistanceMatrix(names, matrix)
    print('Constructed matrix')
    constructor = DistanceTreeConstructor()

    # construct neighbour joining tree
    t = time.time()
    tree = constructor.nj(dm)
    print('Constructed nj tree in {:.4f}s'.format(time.time() - t))
    Phylo.write(tree, "njtree.dnd", "newick")
    remove_inner_nodes_tree("njtree.dnd")

    """
    # construct upgma tree
    t = time.time()
    tree = constructor.upgma(dm)
    print('Constructed upgma tree in {:.4f}s'.format(time.time() - t))
    Phylo.write(tree, "upgmatree.dnd", "newick")
    remove_inner_nodes_tree("upgmatree.dnd")
    """

def ground_truth_guide_trees(dataset):
    _, sequences = torch.max(dataset.sequences, dim=-1)
    distances = dataset.distances
    save_fasta(sequences)
    build_guide_trees(distances)


def approximate_guide_trees(dataset, encoder_model, distance, batch_size, device):
    sys.setrecursionlimit(2400)

    # combine datasets
    # print(dataset['train'].sequences.shape, dataset['val'].sequences.shape, dataset['test'].sequences.shape)
    # strings = torch.cat([dataset['train'].sequences.squeeze(), dataset['val'].sequences.squeeze(), dataset['test'].sequences.squeeze()], dim=0)
    # print(strings.shape)

    # embed sequences and compute distance matrix
    strings_loader = torch.utils.data.DataLoader(dataset.sequences, batch_size=batch_size, shuffle=False)

    t = time.time()
    embedded_strings = embed_strings(strings_loader, encoder_model, device)
    estimate_distances = DISTANCE_MATRIX[distance](embedded_strings, embedded_strings, encoder_model.scaling)
    print('Constructed approximate distance matrix in {:.4f}s'.format(time.time() - t))

    # fix the problems caused by floating point arithmetic: it must be symmetric and with diagonal 0
    estimate_distances = (estimate_distances + estimate_distances.T)/2
    ind = np.diag_indices(estimate_distances.shape[0])
    estimate_distances[ind[0], ind[1]] = 0.0

    # build trees and save files
    _, sequences = torch.max(dataset.sequences, dim=-1)
    save_fasta(sequences)
    build_guide_trees(estimate_distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../edit_distance/data/edit_genome_small.pkl', help='Data path.')
    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        sequences, distances = pickle.load(f)

    dataset = EditDistanceDatasetComplete(sequences['test'], distances['test'])
    ground_truth_guide_trees(dataset)