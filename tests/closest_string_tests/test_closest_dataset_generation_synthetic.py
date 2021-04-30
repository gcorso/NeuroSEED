import unittest
import torch

from closest_string.task.dataset_generator_synthetic import ClosestStringDatasetGenerator
from util.data_handling.string_generator import IndependentGenerator
from edit_distance.task.dataset_generator_synthetic import EditDistanceDatasetGenerator

ALPHABET_SIZE = 6


class TestClosestDatasetGenerationSynthetic(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        self.generator = IndependentGenerator(alphabet_size=ALPHABET_SIZE, seed=0)
        self.dataset = ClosestStringDatasetGenerator(N_reference=10, N_query=15, len_sequence=20, min_changes=3,
                                                     max_changes=10, initials=3, string_generator=self.generator, seed=0)

    def test_shapes(self):
        assert self.dataset.sequences_references.shape == (10, 20), "Sequences references shape is not correct"
        assert self.dataset.sequences_queries.shape == (15, 20), "Sequences queries shape is not correct"
        assert self.dataset.labels.shape == (15, ), "Labels shape is not correct"

    def test_range_labels(self):
        assert torch.all(self.dataset.labels < 10), "Labels out of size references"
        assert torch.all(self.dataset.labels >= 0), "Negative labels"

    def test_range_elements(self):
        assert torch.all(self.dataset.sequences_references < ALPHABET_SIZE), "Sequences references elements out of size"
        assert torch.all(self.dataset.sequences_queries < ALPHABET_SIZE), "Sequences queries elements out of size"
