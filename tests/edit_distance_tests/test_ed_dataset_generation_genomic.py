import random
import unittest
import torch

from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator

def generate_random_dna(length):
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))


class TestEDDatasetGenerationSynthetic(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        random.seed(0)
        strings = [generate_random_dna(50)] + [generate_random_dna(random.randint(10, 50)) for _ in range(39)]
        strings_dict = {'train': strings[:20], 'val': strings[20:30], 'test': strings[30:]}
        self.dataset = EditDistanceGenomicDatasetGenerator(strings=strings_dict)

    def test_shape_sequences(self):
        assert self.dataset.sequences['train'].shape == (20, 50), "Sequences train shape is not correct"
        assert self.dataset.sequences['val'].shape == (10, 50), "Sequences val shape is not correct"
        assert self.dataset.sequences['test'].shape == (10, 50), "Sequences test shape is not correct"

    def test_shape_distances(self):
        assert self.dataset.distances['train'].shape == (20, 20), "Distances train shape is not correct"
        assert self.dataset.distances['val'].shape == (10, 10), "Distances val shape is not correct"
        assert self.dataset.distances['test'].shape == (10, 10), "Distances test shape is not correct"

    def test_range_elements(self):
        assert torch.all(self.dataset.sequences['train'] < 4), "Sequences train elements out of size"
        assert torch.all(self.dataset.sequences['val'] < 4), "Sequences val elements out of size"
        assert torch.all(self.dataset.sequences['test'] < 4), "Sequences test elements out of size"
