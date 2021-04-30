import torch.nn as nn

from hierarchical_clustering.relaxed.models.model import HypHCModel, index_to_one_hot


class HypHCLinear(HypHCModel):
    """ Hyperbolic linear model for hierarchical clustering. """

    def __init__(self, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3, alphabet_size=4,
                 sequence_length=128, device='cpu'):
        super(HypHCLinear, self).__init__(temperature=temperature, init_size=init_size, max_scale=max_scale)
        self.alphabet_size = alphabet_size
        self.device = device
        self.linear = nn.Linear(sequence_length*alphabet_size, rank)

    def encode(self, triple_ids=None, sequences=None):
        sequences = index_to_one_hot(sequences.long(), device=self.device, alphabet_size=self.alphabet_size)
        sequences = sequences.reshape((*sequences.shape[:-2], -1))
        e = self.linear(sequences)
        return e
