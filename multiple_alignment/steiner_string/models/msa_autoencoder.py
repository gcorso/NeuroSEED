from functools import partial

import numpy as np
import torch
import torch.nn as nn

from util.distance_functions.geometric_median import geometric_median
from util.ml_and_math.loss_functions import AverageMeter


class MultipleAlignmentAutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, device, distance, center='geometric_median', normalization=-1):
        super(MultipleAlignmentAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.center = None
        self.center_choice(center, distance)
        self.distance = distance
        self.norm = normalization

    def center_choice(self, center, distance):
        if center == 'geometric_median':
            self.center = partial(geometric_median_torch, distance=distance)
        elif center == 'centroid':
            self.center = centroid
        else:
            raise ValueError

    def normalization(self, enc, hyp_max_norm=1-1e-2):
        if self.norm > 0:
            print(self.norm)
            return self.norm * enc / torch.norm(enc, dim=-1, keepdim=True)
        elif self.distance == 'hyperbolic':
            scaling = (hyp_max_norm / torch.norm(enc, dim=-1, keepdim=True)).clamp_max(1)
            return scaling * enc
        return enc

    def forward(self, sequences):
        (B, K, N) = sequences.shape
        sequences = sequences.reshape(K * B, N)

        # encoder
        enc_sequences = self.encoder(sequences)

        # compute center
        enc_sequences = self.normalization(enc_sequences.reshape(B, K, -1))
        centers, exp_dist = self.center(enc_sequences)
        centers = centers.to(self.device)
        centers = self.normalization(centers.reshape(B, -1))

        # decode
        centers_sequence = self.decoder.decode_and_sample(centers)
        return centers_sequence, exp_dist


def centroid(enc_sequences):
    return torch.mean(enc_sequences, dim=1), -1


def geometric_median_torch(points, distance='euclidean'):
    """ Finds the geometric median via a convex optimization procedure. """
    avg_distance = AverageMeter()

    medians = []
    for Xs in points.detach().cpu().numpy():
        m, val = geometric_median(Xs, distance=distance)
        medians.append(m)
        avg_distance.update(val)

    return torch.from_numpy(np.asarray(medians)).float(), avg_distance.avg
