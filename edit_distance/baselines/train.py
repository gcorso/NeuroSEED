import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from hierarchical_clustering.relaxed.datasets.loading import load_hc_data
from hierarchical_clustering.relaxed.utils.metrics import dasgupta_cost
from hierarchical_clustering.relaxed.utils.tree import to_nx_tree
from util.distance_functions.distance_functions import hyperbolic_distance_numpy
from util.ml_and_math.loss_functions import AverageMeter
from scipy.spatial import distance


def general_arg_parser():
    """ Parsing of parameters common to all the different models """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='Edit distance dataset path')
    parser.add_argument('--distance', type=str, default='euclidean', help='Type of distance to use')
    parser.add_argument('--plot', type=bool, default=False, help='Whether to plot distribution')
    parser.add_argument('--closest_data_path', type=str, default='', help='Closest distance dataset path')
    parser.add_argument('--hierarchical_clustering_path', type=str, default='', help='HC dataset path')
    return parser


def edit_distance_train(encoder_method, vector_distance, texts, distance_labels, plot=False):
    """
    The goal of the training procedure is to find the scaling parameter alpha that minimizes (for i along the dataset)
    \sum_i (r_i - alpha * p_i)^2        where r_i is the real distance and p_i the predicted one.
    Taking partial derivatives it can be shown that alpha = (\sum_i (r_i * p_1)) / (\sum_i p_i ^ 2).
    """
    rp = 0
    p2 = 0
    if plot:
        p = []
        r = []

    for batch in range(len(texts)):
        # embed sequences and compute distances
        v = encoder_method(S=texts[batch])
        p_matrix = distance.cdist(v, v, metric=vector_distance)
        r_matrix = distance_labels[batch]

        rp += np.sum(p_matrix * r_matrix)
        p2 += np.sum(p_matrix * p_matrix)

        if plot:
            p.append(p_matrix)
            r.append(r_matrix)

    if plot:
        p = np.concatenate(p).flatten() * (rp / p2)
        r = np.concatenate(r).flatten()

        plt.plot(p, r, '.', color='black', alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k-', color='r')
        plt.xlabel("Predicted normalized distance", fontsize=12)
        plt.ylabel("Real normalized distance", fontsize=12)
        plt.show()

    return rp / p2


def edit_distance_test(encoder_method, vector_distance, texts, distance_labels, alpha, plot=False):
    loss = AverageMeter(len_tuple=2)
    if plot:
        p = []
        r = []

    for batch in range(len(texts)):
        # embed sequence and compute distances
        v = encoder_method(S=texts[batch])
        p_matrix = distance.cdist(v, v, metric=vector_distance) * alpha
        r_matrix = distance_labels[batch]

        # compute error
        non_zero = np.sum(r_matrix >= 1e-9)
        error = np.abs(p_matrix - r_matrix)
        mse = np.sum(error * error) / non_zero
        mape = np.sum(np.where(r_matrix < 1e-9, 0, error / r_matrix)) / non_zero
        loss.update((mse, mape))

        if plot:
            p.append(p_matrix)
            r.append(r_matrix)

    if plot:
        p = np.concatenate(p).flatten()
        r = np.concatenate(r).flatten()

        plt.plot(p[:3000], r[:3000], '.', color='black', alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k-', color='r')
        plt.xlabel("Predicted normalized distance", fontsize=12)
        plt.ylabel("Real normalized distance", fontsize=12)
        plt.show()

    return loss.avg


def correct_distance(distance):
    # unify nomenclature and distance functions
    if distance == 'hyperbolic':
        distance = hyperbolic_distance_numpy
    elif distance == 'manhattan':
        distance = 'cityblock'
    elif distance == 'square':
        distance = 'sqeuclidean'
    return distance


def test_method(encoder_methods, args):
    """ Trains the baseline and tests it on the various datasets"""
    with open(args.data, 'rb') as f:
        texts, distance_labels = pickle.load(f)

    distance = correct_distance(args.distance)

    normalisation_constant = {}
    for key in texts.keys():
        texts[key] = texts[key].numpy() if len(texts[key].shape) > 2 else texts[key].unsqueeze(0).numpy()
        distance_labels[key] = distance_labels[key].numpy() if len(distance_labels[key].shape) > 2 else distance_labels[
            key].unsqueeze(0).numpy()

        normalisation_constant[key] = texts[key].shape[-1]
        distance_labels[key] = distance_labels[key] / normalisation_constant[key]

    for enc in encoder_methods.keys():
        alpha = edit_distance_train(encoder_method=encoder_methods[enc], vector_distance=distance,
                                    texts=texts['train'], distance_labels=distance_labels['train'], plot=args.plot)

        val_mse, val_mape = edit_distance_test(encoder_method=encoder_methods[enc], vector_distance=distance, texts=texts['val'],
                                               distance_labels=distance_labels['val'], alpha=alpha, plot=args.plot)

        test_mse, test_mape = edit_distance_test(encoder_method=encoder_methods[enc], vector_distance=distance,
                                                 texts=texts['test'], distance_labels=distance_labels['test'],
                                                 alpha=alpha, plot=args.plot)

        print("{} results val: mse {:.6f}  mape {:.4f}  test: {:.7f}, {:.4f}"
              .format(enc, val_mse, val_mape, test_mse, test_mape))

        if args.closest_data_path != '':
            closest_string_test(encoder_method=encoder_methods[enc], vector_distance=distance,
                                data_path=args.closest_data_path)

        if args.hierarchical_clustering_path != '':
            hierarchical_clustering_test(encoder_method=encoder_methods[enc], vector_distance=distance,
                                         data_path=args.hierarchical_clustering_path, alpha=alpha)


def closest_string_test(encoder_method, vector_distance, data_path, slots=10):
    # load dataset as numpy arrays
    with open(data_path, 'rb') as f:
        texts_references, texts_queries, labels = pickle.load(f)
    texts_references, texts_queries, labels = texts_references.numpy(), texts_queries.numpy(), labels.long().numpy()

    # encode sequences and compute distance matrix
    rq = encoder_method(S=np.concatenate([texts_references, texts_queries], axis=0))
    r = rq[:len(texts_references)]
    q = rq[len(texts_references):]
    distance_matrix = distance.cdist(r, q, metric=vector_distance)

    # evaluate performance based on rank of correct answer
    label_distances = distance_matrix[labels, np.arange(0, distance_matrix.shape[1])]
    rank = np.sum(distance_matrix <= label_distances.reshape(1, *(label_distances.shape)), axis=0)
    acc = [np.mean(rank <= i + 1) for i in range(slots)]

    print('Results: accuracy {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(*acc))
    print('Top1: {:.3f}  Top5: {:.3f}  Top10: {:.3f}'.format(acc[0], acc[4], acc[9]))


def hierarchical_clustering_test(encoder_method, vector_distance, data_path, alpha):
    # load dataset
    strings, similarities = load_hc_data(data_path)

    # embed sequence and compute distances
    embedded_texts = encoder_method(S=strings)
    estimate_distances = distance.cdist(embedded_texts, embedded_texts, metric=vector_distance) * alpha

    # fix the problems caused by floating point arithmetic: it must be symmetric and with diagonal 0
    estimate_distances = (estimate_distances + estimate_distances.T) / 2
    ind = np.diag_indices(estimate_distances.shape[0])
    estimate_distances[ind[0], ind[1]] = 0.0

    # perform agglomerative clustering and compute Dasgupta's cost
    metrics = {}
    for method in ["single", "complete", "average", "ward"]:
        metrics[method] = {}
        baseline_tree = to_nx_tree(linkage(squareform(estimate_distances), method))
        dc = dasgupta_cost(baseline_tree, similarities)
        metrics[method]["DC"] = dc
    print(metrics)
