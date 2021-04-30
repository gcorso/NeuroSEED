import random
import unittest

from multiple_alignment.steiner_string.task.dataset_generator_genome import MSAPairDatasetGeneratorGenome
from tests.edit_distance_tests.test_ed_dataset_generation_genomic import generate_random_dna


class TestMSADatasetGenerationGenomic(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        random.seed(0)
        strings = [generate_random_dna(50)] + [generate_random_dna(random.randint(10, 50)) for _ in range(39)]
        sequences = {
            'train': strings[:10],
            'val': strings[10:15],
            'val_msa': [strings[15:20], strings[20:25]],
            'test': [strings[25:30], strings[30:35], strings[35:]]
        }
        self.dataset = MSAPairDatasetGeneratorGenome(strings=sequences, length=50)

    def test_sequences_shapes(self):
        assert self.dataset.datasets['train']['texts'].shape == (10, 50), "Sequences train shape is not correct"
        assert self.dataset.datasets['val']['texts'].shape == (5, 50), "Sequences val shape is not correct"
        assert self.dataset.datasets['val_msa']['texts'].shape == (2, 5, 50), "Sequences val msa shape is not correct"
        assert self.dataset.datasets['test']['texts'].shape == (3, 5, 50), "Sequences test shape is not correct"

    def test_distances_shapes(self):
        assert self.dataset.datasets['train']['distances'].shape == (10, 10), "Distances train shape is not correct"
        assert self.dataset.datasets['val']['distances'].shape == (5, 5), "Distances val shape is not correct"