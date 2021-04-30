import torch.nn as nn

from util.ml_and_math.layers import MLP


class GRU(nn.Module):

    def __init__(self, len_sequence, embedding_size, hidden_size, recurrent_layers, device, readout_layers, alphabet_size=4,
                 dropout=0.0):
        super(GRU, self).__init__()

        self.len_sequence = len_sequence
        self.sequence_encoder = nn.GRU(input_size=alphabet_size, hidden_size=hidden_size, num_layers=recurrent_layers,
                                   dropout=dropout)
        self.readout = MLP(in_size=hidden_size, hidden_size=hidden_size, out_size=embedding_size,
                           layers=readout_layers, mid_activation='relu', dropout=dropout, device=device)
        self.to(device)

    def forward(self, sequence):
        # sequence (B, N, 4)
        (B, N, _) = sequence.shape

        # apply recurrent layers
        sequence = sequence.transpose(0, 1)
        enc_sequence, _ = self.sequence_encoder(sequence)
        enc_sequence = enc_sequence[-1]

        # apply readout
        enc_sequence = self.readout(enc_sequence)
        return enc_sequence
