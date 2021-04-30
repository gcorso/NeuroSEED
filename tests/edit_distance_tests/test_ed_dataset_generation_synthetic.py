import unittest
import torch

from util.data_handling.string_generator import IndependentGenerator
from edit_distance.task.dataset_generator_synthetic import EditDistanceDatasetGenerator

ALPHABET_SIZE = 6


class TestEDDatasetGenerationSynthetic(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        self.generator = IndependentGenerator(alphabet_size=ALPHABET_SIZE, seed=0)
        self.dataset = EditDistanceDatasetGenerator(
            N_batches={"train": 4, "val": 2, "test": 3},
            batch_size={"train": 5, "val": 3, "test": 4},
            len_sequence={"train": 10, "val": 10, "test": 10},
            max_changes={"train": 4, "val": 4, "test": 4},
            string_generator=self.generator, seed=0)

    def test_shape_sequences(self):
        assert self.dataset.sequences['train'].shape == (4, 5, 10), "Sequences train shape is not correct"
        assert self.dataset.sequences['val'].shape == (2, 3, 10), "Sequences val shape is not correct"
        assert self.dataset.sequences['test'].shape == (3, 4, 10), "Sequences test shape is not correct"

    def test_shape_distances(self):
        assert self.dataset.distances['train'].shape == (4, 5, 5), "Distances train shape is not correct"
        assert self.dataset.distances['val'].shape == (2, 3, 3), "Distances val shape is not correct"
        assert self.dataset.distances['test'].shape == (3, 4, 4), "Distances test shape is not correct"

    def test_range_elements(self):
        assert torch.all(self.dataset.sequences['train'] < ALPHABET_SIZE), "Sequences train elements out of size"
        assert torch.all(self.dataset.sequences['val'] < ALPHABET_SIZE), "Sequences val elements out of size"
        assert torch.all( self.dataset.sequences['test'] < ALPHABET_SIZE), "Sequences test elements out of size"
