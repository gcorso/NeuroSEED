import unittest
import torch

from util.distance_functions.distance_functions import DISTANCE_TORCH

SIZE = (3, 4, 5, 6, 7)

class TestDistanceTorchFunctions(unittest.TestCase):

    def random_vector_unit_ball(self, size, epsilon=1e-5):
        vec = torch.rand(size)
        norm = torch.norm(vec) + epsilon
        vec = vec / norm.clamp_min(1)
        return vec

    def test_shape_random_points(self):
        torch.manual_seed(0)

        vec_1 = self.random_vector_unit_ball(SIZE)
        vec_2 = self.random_vector_unit_ball(SIZE)

        for d in DISTANCE_TORCH.keys():
            dist = DISTANCE_TORCH[d](vec_1, vec_2)

            assert dist.shape == SIZE[:-1], "Wrong shape for " + d + " distance"

    def test_identity_random_points(self):
        torch.manual_seed(1)

        vec_1 = self.random_vector_unit_ball(SIZE)

        for d in DISTANCE_TORCH.keys():
            dist = DISTANCE_TORCH[d](vec_1, vec_1)

            assert torch.all(torch.isclose(dist, torch.tensor(0.0), atol=1e-6)), \
                d + ' distance of vector with itself different from 0'

    def test_reflection_random_points(self):
        torch.manual_seed(2)

        vec_1 = self.random_vector_unit_ball(SIZE)
        vec_2 = self.random_vector_unit_ball(SIZE)

        for d in DISTANCE_TORCH.keys():
            dist1 = DISTANCE_TORCH[d](vec_1, vec_2)
            dist2 = DISTANCE_TORCH[d](vec_2, vec_1)

            assert torch.all(torch.isclose(dist1, dist2, atol=1e-6)), \
                d + ' distance of vectors is not symmetric (commutative)'
