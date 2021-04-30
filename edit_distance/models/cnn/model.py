import torch
import torch.nn as nn

from util.ml_and_math.layers import MLP


class CNN(nn.Module):

    def __init__(self, len_sequence, embedding_size, channels, device, readout_layers=2, layers=3, dropout=0.,
                 kernel_size=3, alphabet_size=4, pooling='avg', non_linearity=True, batch_norm=False, stride=1):
        super(CNN, self).__init__()
        assert pooling == 'none' or pooling == 'avg' or pooling == 'max', "Wrong pooling type"

        self.layers = layers
        self.kernel_size = kernel_size
        self.embedding = nn.Linear(alphabet_size, channels)

        # construct convolutional layers
        self.conv = torch.nn.Sequential()
        for l in range(self.layers):
            self.conv.add_module('conv_' + str(l + 1),
                                 nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                           padding=kernel_size // 2, stride=stride))
            len_sequence = (len_sequence - 1) // stride + 1
            if batch_norm:
                self.conv.add_module('batchnorm_' + str(l + 1), nn.BatchNorm1d(num_features=channels))
            if non_linearity:
                self.conv.add_module('relu_' + str(l + 1), nn.ReLU())

            if pooling == 'avg':
                self.conv.add_module(pooling + '_pool_' + str(l + 1), nn.AvgPool1d(2))
                len_sequence //= 2
            elif pooling == 'max':
                self.conv.add_module(pooling + 'pool_' + str(l + 1), nn.MaxPool1d(2))
                len_sequence //= 2

        # construct readout
        print(len_sequence)
        flat_size = channels * len_sequence
        self.readout = MLP(in_size=flat_size, hidden_size=embedding_size, out_size=embedding_size,
                           layers=readout_layers, mid_activation='relu', dropout=dropout, device=device)
        self.to(device)

    def forward(self, sequence):
        (B, N, _) = sequence.shape

        # initial transformation
        sequence = self.embedding(sequence)

        # apply convolutions
        enc_sequence = sequence.transpose(1, 2)
        enc_sequence = self.conv(enc_sequence)

        # flatten and apply readout layer
        enc_sequence = enc_sequence.reshape(B, -1)
        embedding = self.readout(enc_sequence)
        return embedding
