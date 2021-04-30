import unittest
import torch
import numpy as np

from closest_string.task.dataset_generator_synthetic import ClosestStringDatasetGenerator
from hierarchical_clustering.task.dataset_generator_synthetic import HierarchicalClusteringDatasetGenerator
from util.data_handling.string_generator import IndependentGenerator
from edit_distance.task.dataset_generator_synthetic import EditDistanceDatasetGenerator

ALPHABET_SIZE = 6


class TestHCDatasetGenerationSynthetic(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        self.generator = IndependentGenerator(alphabet_size=ALPHABET_SIZE, seed=0)
        self.dataset = HierarchicalClusteringDatasetGenerator(N_reference=10, N_leaves=15, len_sequence=20,
                                                              min_changes=3, max_changes=10,
                                                              string_generator=self.generator, seed=0)

    def test_shapes(self):
        assert self.dataset.sequences_leaves.shape == (15, 20), "Sequences shape is not correct"
        assert self.dataset.similarities.shape == (15, 15), "Similarities shape is not correct"

    def test_range_labels(self):
        assert np.all(self.dataset.similarities <= 1), "Similarities out of size references"
        assert np.all(self.dataset.similarities >= 0), "Negative similarities"

    def test_range_elements(self):
        assert np.all(self.dataset.sequences_leaves < ALPHABET_SIZE), "Sequences elements out of size"
