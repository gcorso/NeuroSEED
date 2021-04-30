import random
import unittest
import torch
import numpy as np

from hierarchical_clustering.task.dataset_generator_genomic import HierarchicalClusteringGenomicDatasetGenerator


def generate_random_dna(length):
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))


class TestHCDatasetGenerationSynthetic(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        random.seed(0)
        strings = [generate_random_dna(50)] + [generate_random_dna(random.randint(10, 50)) for _ in range(19)]
        self.dataset = HierarchicalClusteringGenomicDatasetGenerator(strings=strings)

    def test_shapes(self):
        assert self.dataset.sequences_leaves.shape == (20, 50), "Sequences shape is not correct"
        assert self.dataset.similarities.shape == (20, 20), "Similarities shape is not correct"

    def test_range_labels(self):
        assert np.all(self.dataset.similarities <= 1), "Similarities out of size references"
        assert np.all(self.dataset.similarities >= 0), "Negative similarities"
