import torch
import torch.nn as nn

from util.data_handling.data_loader import index_to_one_hot
from util.ml_and_math.layers import MLP


class MLPEncoder(nn.Module):

    def __init__(self, len_sequence, embedding_size, layers, hidden_size, device, dropout=0., alphabet_size=4):
        super(MLPEncoder, self).__init__()

        self.device = device
        self.mlp = MLP(in_size=len_sequence * alphabet_size, hidden_size=hidden_size, out_size=embedding_size,
                           layers=layers, mid_activation='relu', dropout=dropout, device=device)
        self.to(device)

    def forward(self, sequence):
        (B, N) = sequence.shape
        sequence = index_to_one_hot(sequence, device=self.device)
        embedding = self.mlp(sequence.reshape(B, -1))
        return embedding


class MLPDecoder(nn.Module):

    def __init__(self, len_sequence, embedding_size, layers, hidden_size, device, dropout=0., alphabet_size=4):
        super(MLPDecoder, self).__init__()

        self.len_sequence = len_sequence
        self.mlp = MLP(in_size=embedding_size, hidden_size=hidden_size, out_size=len_sequence * (alphabet_size+1),
                       layers=layers, mid_activation='relu', dropout=dropout, device=device)
        self.to(device)

    def forward(self, enc):
        # enc (B, e)
        (B, _) = enc.shape
        dec_sequence = self.mlp(enc)
        dec_sequence = dec_sequence.reshape(B, self.len_sequence, -1)
        return dec_sequence

    def decode_and_sample(self, enc):
        dec_sequence = self.forward(enc)
        dec_sequence = torch.argmax(dec_sequence, dim=-1)
        return dec_sequence