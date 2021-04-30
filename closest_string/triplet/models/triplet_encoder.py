import torch
import torch.nn as nn
import torch.nn.functional as F

from util.distance_functions.distance_functions import DISTANCE_TORCH


class TripletEncoder(nn.Module):

    def __init__(self, embedding_model, distance='euclidean', scaling=False):
        super(TripletEncoder, self).__init__()

        self.embedding_model = embedding_model
        self.distance = DISTANCE_TORCH[distance]
        self.distance_str = distance

        self.scaling = None
        if scaling:
            self.scaling = True
            self.radius = nn.Parameter(torch.Tensor([0.5]), requires_grad=False)

    def normalize_embeddings(self, embeddings):
        """ Project embeddings to an hypersphere of a certain radius """
        min_scale = 1e-7

        if self.distance_str == 'hyperbolic':
            max_scale = 1 - 1e-3
        else:
            max_scale = 1e10

        return F.normalize(embeddings, p=2, dim=1) * self.radius.clamp_min(min_scale).clamp_max(max_scale)

    def encode(self, sequence):
        """ Use embedding model and normalization to encode some sequences. """
        enc_sequence = self.embedding_model(sequence)
        if self.scaling is not None:
            enc_sequence = self.normalize_embeddings(enc_sequence)
        return enc_sequence

    def forward(self, sequence):
        # flatten couples
        (B, _, N, _) = sequence.shape
        sequence = sequence.reshape(3 * B, N, -1)

        # encode sequences
        enc_sequence = self.encode(sequence)

        # compute distances
        enc_sequence = enc_sequence.reshape(B, 3, -1)
        distance_positive = self.distance(enc_sequence[:, 0], enc_sequence[:, 1])
        distance_negative = self.distance(enc_sequence[:, 0], enc_sequence[:, 2])

        return distance_positive, distance_negative


def triplet_loss(distance_positive, distance_negative, device, margin=0.05):
    zero = torch.tensor([0.0]).to(device)
    margins = torch.max(distance_positive - distance_negative + margin, zero)
    return torch.mean(margins)