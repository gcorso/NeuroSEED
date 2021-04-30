import torch
import torch.nn as nn
from util.ml_and_math.layers import MLP


class CNNEncoder(nn.Module):

    def __init__(self, len_sequence, embedding_size, channels, device, readout_layers=2, layers=3, dropout=0., kernel_size=3,
                 alphabet_size=4, non_linearity=True):
        super(CNNEncoder, self).__init__()

        self.embedding = nn.Embedding(alphabet_size+1, channels)
        self.conv = torch.nn.Sequential()
        for l in range(layers):
            self.conv.add_module('conv_' + str(l+1), nn.Conv1d(in_channels=channels, out_channels=channels,
                                                               kernel_size=kernel_size, padding=kernel_size // 2))
            if non_linearity:
                self.conv.add_module('relu_' + str(l + 1), nn.ReLU())

        flat_size = channels * len_sequence
        self.readout = MLP(in_size=flat_size, hidden_size=embedding_size, out_size=embedding_size,
                           layers=readout_layers, mid_activation='relu', dropout=dropout, device=device)
        self.to(device)

    def forward(self, sequence):
        (B, N) = sequence.shape
        sequence = self.embedding(sequence)
        enc_sequence = sequence.transpose(1, 2)
        enc_sequence = self.conv(enc_sequence)
        enc_sequence = enc_sequence.reshape(B, -1)
        embedding = self.readout(enc_sequence)
        return embedding


class CNNDecoder(nn.Module):

    def __init__(self, len_sequence, embedding_size, channels, device, readout_layers=2, layers=3, dropout=0., kernel_size=3,
                 alphabet_size=4, non_linearity=True):
        super(CNNDecoder, self).__init__()

        self.len_sequence = len_sequence
        flat_size = channels * len_sequence
        self.de_readout = MLP(in_size=embedding_size, hidden_size=embedding_size, out_size=flat_size,
                           layers=readout_layers, mid_activation='relu', dropout=dropout, device=device)
        self.de_conv = torch.nn.Sequential()
        for l in range(layers):
            self.de_conv.add_module('de_conv_' + str(l+1), nn.ConvTranspose1d(in_channels=channels, out_channels=channels,
                                                               kernel_size=kernel_size, padding=kernel_size // 2))
            if non_linearity:
                self.de_conv.add_module('relu_' + str(l + 1), nn.ReLU())

        self.de_embedding = nn.Linear(in_features=channels, out_features=alphabet_size+1)
        self.to(device)

    def forward(self, enc):
        # enc (B, e)
        (B, _) = enc.shape
        enc_sequence = self.de_readout(enc)
        enc_sequence = enc_sequence.reshape(B, -1, self.len_sequence)
        enc_sequence = self.de_conv(enc_sequence)
        enc_sequence = enc_sequence.transpose(1, 2)
        dec_sequence = self.de_embedding(enc_sequence)
        return dec_sequence

    def decode_and_sample(self, enc):
        dec_sequence = self.forward(enc)
        dec_sequence = torch.argmax(dec_sequence, dim=-1)
        return dec_sequence