from hierarchical_clustering.relaxed.models.model import HypHCModel, index_to_one_hot
from util.ml_and_math.layers import MLP


class HypHCMLP(HypHCModel):
    """ Hyperbolic MLP model for hierarchical clustering. """

    def __init__(self, layers, hidden_size, dropout=0.0, batch_norm=False, rank=2, temperature=0.05,
                 init_size=1e-3, max_scale=1. - 1e-3, alphabet_size=4, sequence_length=128, device='cpu'):
        super(HypHCMLP, self).__init__(temperature=temperature, init_size=init_size, max_scale=max_scale)

        self.alphabet_size = alphabet_size
        self.device = device
        self.encoder = MLP(in_size=alphabet_size * sequence_length, hidden_size=hidden_size, out_size=rank,
                           layers=layers, mid_activation='relu', dropout=dropout, device=device, mid_b_norm=batch_norm)

    def encode(self, triple_ids=None, sequences=None):
        sequences = index_to_one_hot(sequences.long(), device=self.device)
        sequences = sequences.reshape((*sequences.shape[:-2], -1))
        e = self.encoder(sequences)
        return e
