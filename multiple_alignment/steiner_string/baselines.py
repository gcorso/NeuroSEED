import pickle
import numpy as np
import Levenshtein

from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix
from multiple_alignment.steiner_string.task.dataset import MultipleAlignmentDataset
from multiple_alignment.steiner_string.models.loss import torch_to_string
from util.ml_and_math.loss_functions import AverageMeter


def multiple_alignment_cost_baseline(sequences, alphabet_size=4):
    (B, K, N) = sequences.shape

    avg_distance = AverageMeter()
    min_distance = AverageMeter()
    median_distance = AverageMeter()
    quickmedian_distance = AverageMeter()

    for i in range(B):
        strings = torch_to_string(sequences[i], alphabet_size)
        distances = cross_distance_matrix(strings, strings)

        # Average distance between sequences
        avg_distance.update(np.mean(distances))

        # Center string
        min_distance.update(np.min(np.mean(distances, axis=1)))

        # Greedy median algorithm (Kruzslicz 1999)
        median = Levenshtein.median(strings)
        d_median = cross_distance_matrix([median], strings)
        median_distance.update(np.mean(d_median))

        # Greedy median algorithm (Casacuberta & Antonio 1997)
        quickmedian = Levenshtein.quickmedian(strings)
        d_quickmedian = cross_distance_matrix([quickmedian], strings)
        quickmedian_distance.update(np.mean(d_quickmedian))

    return avg_distance.avg, min_distance.avg, median_distance.avg, quickmedian_distance.avg


def run_baselines(dataset_path):
    with open(dataset_path, 'rb') as f:
        datasets_raw = pickle.load(f)

    # validation MSA dataset
    print("MSA val")
    dataset = MultipleAlignmentDataset(datasets_raw['val_msa']['sequences'])
    cost = multiple_alignment_cost_baseline(dataset.sequences)
    print("Baselines costs for multiple alignment: average {:.4f},  minimum {:.4f},  "
          "median {:.4f},  quickmedian {:.4f}".format(*cost))

    # test dataset
    print("MSA test")
    dataset = MultipleAlignmentDataset(datasets_raw['test']['sequences'])
    cost = multiple_alignment_cost_baseline(dataset.sequences)
    print("Baselines costs for multiple alignment: average {:.4f},  minimum {:.4f},  "
          "median {:.4f},  quickmedian {:.4f}".format(*cost))


run_baselines("../data/msa_qiita_large.pkl")