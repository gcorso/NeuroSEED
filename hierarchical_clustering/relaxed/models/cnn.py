import torch
import torch.nn as nn

from hierarchical_clustering.relaxed.models.model import HypHCModel, index_to_one_hot
from util.ml_and_math.layers import MLP


class HypHCCNN(HypHCModel):
    """ Hyperbolic convolutional model for hierarchical clustering. """

    def __init__(self, layers, channels, kernel_size, readout_layers=2, non_linearity=True, pooling='avg', dropout=0.0,
                 batch_norm=False, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3, alphabet_size=4,
                 sequence_length=128, device='cpu'):
        super(HypHCCNN, self).__init__(temperature=temperature, init_size=init_size, max_scale=max_scale)

        self.alphabet_size = alphabet_size
        self.device = device
        self.embedding = nn.Linear(alphabet_size, channels)
        self.conv = torch.nn.Sequential()
        for l in range(layers):
            self.conv.add_module('conv_' + str(l + 1), nn.Conv1d(in_channels=channels, out_channels=channels,
                                                                 kernel_size=kernel_size, padding=kernel_size // 2))
            if batch_norm:
                self.conv.add_module('batchnorm_' + str(l + 1), nn.BatchNorm1d(num_features=channels))
            if non_linearity:
                self.conv.add_module('relu_' + str(l + 1), nn.ReLU())

            if pooling == 'avg':
                self.conv.add_module(pooling + '_pool_' + str(l + 1), nn.AvgPool1d(2))
            elif pooling == 'max':
                self.conv.add_module(pooling + 'pool_' + str(l + 1), nn.MaxPool1d(2))

        flat_size = channels * sequence_length if pooling == 'none' else channels * (sequence_length // 2 ** layers)
        self.readout = MLP(in_size=flat_size, hidden_size=rank, out_size=rank,
                           layers=readout_layers, mid_activation='relu', dropout=dropout, device=device)

    def encode(self, triple_ids=None, sequences=None):
        (B, N) = sequences.shape
        sequences = index_to_one_hot(sequences.long(), device=self.device)
        sequences = self.embedding(sequences)
        enc_sequences = sequences.transpose(1, 2)
        enc_sequences = self.conv(enc_sequences)
        enc_sequences = enc_sequences.reshape(B, -1)
        embedding = self.readout(enc_sequences)
        return embedding


class CNN(nn.Module):

    def __init__(self, len_text, embedding_size, channels, device, readout_layers=2, layers=3, dropout=0.,
                 kernel_size=3, alphabet_size=4, pooling='avg', non_linearity=True, batch_norm=False):
        super(CNN, self).__init__()
        assert pooling == 'none' or pooling == 'avg' or pooling == 'max', "Wrong pooling type"

        self.len_text = len_text
        self.layers = layers
        self.kernel_size = kernel_size
        self.embedding = nn.Linear(alphabet_size, channels)
        self.conv = torch.nn.Sequential()
        for l in range(self.layers):
            self.conv.add_module('conv_' + str(l + 1), nn.Conv1d(in_channels=channels, out_channels=channels,
                                                                 kernel_size=kernel_size, padding=kernel_size // 2))
            if batch_norm:
                self.conv.add_module('batchnorm_' + str(l + 1), nn.BatchNorm1d(num_features=channels))
            if non_linearity:
                self.conv.add_module('relu_' + str(l + 1), nn.ReLU())

            if pooling == 'avg':
                self.conv.add_module(pooling + '_pool_' + str(l + 1), nn.AvgPool1d(2))
            elif pooling == 'max':
                self.conv.add_module(pooling + 'pool_' + str(l + 1), nn.MaxPool1d(2))

        flat_size = channels * len_text if pooling == 'none' else channels * (len_text // 2 ** layers)
        self.readout = MLP(in_size=flat_size, hidden_size=embedding_size, out_size=embedding_size,
                           layers=readout_layers, mid_activation='relu', dropout=dropout, device=device)
        self.to(device)

    def forward(self, text):
        (B, N, _) = text.shape
        text = self.embedding(text)
        enc_text = text.transpose(1, 2)
        enc_text = self.conv(enc_text)
        enc_text = enc_text.reshape(B, -1)
        embedding = self.readout(enc_text)
        return embedding
