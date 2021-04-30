from functools import partial

import torch
import torch.nn as nn
import numpy as np


class Kmer(nn.Module):

    def __init__(self, len_sequence, embedding_size, device, k, dropout=0.0, alphabet_size=4):
        super(Kmer, self).__init__()
        self.alphabet_size = alphabet_size
        self.k = k
        self.device = device

    def forward(self, sequence):
        # apply kernel to find kmers
        sequence = torch.argmax(sequence, dim=-1).detach().numpy()
        kernel = [self.alphabet_size ** p for p in range(self.k)]
        kmers = np.apply_along_axis(partial(np.convolve, v=kernel, mode='valid'), 1, sequence)

        # count kmers
        vectors = np.zeros((sequence.shape[0], self.alphabet_size ** self.k))
        for d in range(len(sequence)):
            bbins = np.bincount(kmers[d])
            vectors[d][:len(bbins)] += bbins

        return torch.from_numpy(vectors).float().to(self.device)
