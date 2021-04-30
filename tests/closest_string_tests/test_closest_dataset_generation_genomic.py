import random
import unittest
import torch

from closest_string.task.dataset_generator_genomic import ClosestStringGenomicDatasetGenerator


def generate_random_dna(length):
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))


class TestClosestDatasetGenerationSynthetic(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        random.seed(0)
        references = [generate_random_dna(50)] + [generate_random_dna(random.randint(10, 50)) for _ in range(9)]
        queries = [generate_random_dna(random.randint(10, 50)) for _ in range(25)]
        self.dataset = ClosestStringGenomicDatasetGenerator(strings_reference=references, strings_query=queries, n_queries=15)

    def test_shapes(self):
        assert self.dataset.sequences_references.shape == (10, 50), "Sequences references shape is not correct"
        assert self.dataset.sequences_queries.shape == (15, 50), "Sequences queries shape is not correct"
        assert self.dataset.labels.shape == (15, ), "Labels shape is not correct"

    def test_range_labels(self):
        assert torch.all(self.dataset.labels < 10), "Labels out of size references"
        assert torch.all(self.dataset.labels >= 0), "Negative labels"
