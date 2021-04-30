import random
import unittest
import Levenshtein

from util.bioinformatics_algorithms.edit_distance import dp_edit_distance
from util.data_handling.string_generator import IndependentGenerator

TEST_CASES = 50


class TestEditDistanceConsensus(unittest.TestCase):

    def test_100_independent(self):
        for seed in range(TEST_CASES):
            generator = IndependentGenerator(alphabet_size=4, seed=seed)
            sequence1 = generator.generate(length=100)
            sequence2 = generator.generate(length=100)

            distance1 = dp_edit_distance(sequence1, sequence2)

            string1 = "".join(chr(s + 97) for s in sequence1)
            string2 = "".join(chr(s + 97) for s in sequence2)
            distance2 = Levenshtein.distance(string1, string2)

            assert distance1 == distance2, \
                'Mismatch between dp_edit_distance and Levenshtein.distance'

    def test_random_len_independent(self):
        for seed in range(TEST_CASES):
            generator = IndependentGenerator(alphabet_size=4, seed=seed)
            random.seed(seed)

            sequence1 = generator.generate(length=random.randint(10, 100))
            sequence2 = generator.generate(length=random.randint(10, 100))

            distance1 = dp_edit_distance(sequence1, sequence2)

            string1 = "".join(chr(s + 97) for s in sequence1)
            string2 = "".join(chr(s + 97) for s in sequence2)
            distance2 = Levenshtein.distance(string1, string2)

            assert distance1 == distance2, \
                'Mismatch between dp_edit_distance and Levenshtein.distance'

    def test_equal_independent(self):
        for seed in range(TEST_CASES):
            generator = IndependentGenerator(alphabet_size=4, seed=seed)
            random.seed(seed)

            sequence1 = generator.generate(length=random.randint(10, 100))
            distance1 = dp_edit_distance(sequence1, sequence1)

            string1 = "".join(chr(s + 97) for s in sequence1)
            distance2 = Levenshtein.distance(string1, string1)

            assert distance1 == 0, 'dp_edit_distance on equal sequences different from 0'
            assert distance2 == 0, 'Levenshtein.distance on equal sequences different from 0'
