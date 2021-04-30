"""
    Part of this code was adapted from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from hierarchical_clustering.relaxed.utils.metrics import dasgupta_cost
from hierarchical_clustering.relaxed.utils.tree import to_nx_tree
from util.distance_functions.distance_matrix import DISTANCE_MATRIX
from closest_string.test import embed_strings
from hierarchical_clustering.relaxed.datasets.hc_dataset import load_hc_data
from util.data_handling.data_loader import index_to_one_hot


def execute_test(args):
    # set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'
    print('Using device:', device)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # model
    model_class, model_args, state_dict, distance = torch.load(args.encoder_path)
    encoder_model = model_class(**vars(model_args))

    # Restore best model
    print('Loading model ' + args.encoder_path)
    encoder_model.load_state_dict(state_dict)
    encoder_model.eval()

    hierarchical_clustering_testing(encoder_model, args.data, args.batch_size, device, distance)


def hierarchical_clustering_testing(encoder_model, data_path, batch_size, device, distance):
    # load data
    strings, similarities = load_hc_data(data_path)
    strings = torch.from_numpy(strings).long()
    print("Hierarchical", strings.shape)
    strings = index_to_one_hot(strings)
    strings_loader = torch.utils.data.DataLoader(strings, batch_size=batch_size, shuffle=False)

    # embed sequences and compute distance matrix
    embedded_strings = embed_strings(strings_loader, encoder_model, device)
    estimate_distances = DISTANCE_MATRIX[distance](embedded_strings, embedded_strings, encoder_model.scaling)

    # fix the problems caused by floating point arithmetic: it must be symmetric and with diagonal 0
    estimate_distances = (estimate_distances + estimate_distances.T)/2
    ind = np.diag_indices(estimate_distances.shape[0])
    estimate_distances[ind[0], ind[1]] = 0.0

    # run agglomerative clustering algorithms
    metrics = {}
    for method in ["single", "complete", "average", "ward"]:
        metrics[method] = {}
        baseline_tree = to_nx_tree(linkage(squareform(estimate_distances), method))
        dc = dasgupta_cost(baseline_tree, similarities)
        metrics[method]["DC"] = dc
    print(metrics)
