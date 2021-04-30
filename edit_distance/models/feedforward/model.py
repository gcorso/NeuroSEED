import torch.nn as nn

from util.ml_and_math.layers import MLP


class MLPEncoder(nn.Module):

    def __init__(self, len_sequence, embedding_size, layers, hidden_size, device, batch_norm=True, dropout=0.,
                 alphabet_size=4):
        super(MLPEncoder, self).__init__()

        self.encoder = MLP(in_size=alphabet_size * len_sequence, hidden_size=hidden_size, out_size=embedding_size,
                           layers=layers, mid_activation='relu', dropout=dropout, device=device, mid_b_norm=batch_norm)

    def forward(self, sequence):
        # flatten sequence
        B = sequence.shape[0]
        sequence = sequence.reshape(B, -1)

        # apply MLP
        emb = self.encoder(sequence)
        return emb
