import random
import unittest
from util.data_handling.string_generator import IndependentGenerator
from util.bioinformatics_algorithms.exact_matching import brute_force, z_matching, boyer_moore_matching, kmp_matching

TEST_CASES = 100
EXACT_MATCHING_ALGORITHMS = {
    'brute_force': brute_force,
    'z_matching': z_matching,
    'boyer_moore': boyer_moore_matching,
    'knuth_morris_pratt': kmp_matching
}


class TestExactMatchingConsensus(unittest.TestCase):

    def test_10independent_5substr(self):
        for seed in range(TEST_CASES):
            generator = IndependentGenerator(alphabet_size=4, seed=seed)
            sequence = generator.generate(length=10)

            random.seed(seed)
            start = random.randint(0, 4)
            pattern = sequence[start:start + 5]

            matches = {}
            for algorithm in EXACT_MATCHING_ALGORITHMS.keys():
                matches[algorithm] = sorted(EXACT_MATCHING_ALGORITHMS[algorithm](pattern, sequence))

            for algorithm in EXACT_MATCHING_ALGORITHMS.keys():
                if algorithm != 'brute_force':
                    assert matches[algorithm] == matches['brute_force'], \
                        'Mismatch between brute force and ' + algorithm + ": " + str(matches)

    def test_100independent_10substr(self):
        for seed in range(TEST_CASES):
            generator = IndependentGenerator(alphabet_size=4, seed=seed)
            sequence = generator.generate(length=100)

            random.seed(seed)
            start = random.randint(0, 89)
            pattern = sequence[start:start + 10]

            matches = {}
            for algorithm in EXACT_MATCHING_ALGORITHMS.keys():
                matches[algorithm] = sorted(EXACT_MATCHING_ALGORITHMS[algorithm](pattern, sequence))

            for algorithm in EXACT_MATCHING_ALGORITHMS.keys():
                if algorithm != 'brute_force':
                    assert matches[algorithm] == matches['brute_force'], \
                        'Mismatch between brute force and ' + algorithm + ": " + str(matches)

    def test_100independent_5independent(self):
        for seed in range(TEST_CASES):
            generator = IndependentGenerator(alphabet_size=4, seed=seed)
            sequence = generator.generate(length=100)
            pattern = generator.generate(length=5)

            random.seed(seed)
            matches = {}
            for algorithm in EXACT_MATCHING_ALGORITHMS.keys():
                matches[algorithm] = sorted(EXACT_MATCHING_ALGORITHMS[algorithm](pattern, sequence))

            for algorithm in EXACT_MATCHING_ALGORITHMS.keys():
                if algorithm != 'brute_force':
                    assert matches[algorithm] == matches['brute_force'], \
                        'Mismatch between brute force and ' + algorithm + ": " + str(matches)
