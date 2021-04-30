import torch
import torch.nn as nn
from geoopt import PoincareBall

from util.distance_functions.distance_functions import DISTANCE_TORCH


class PairDistanceAutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, distance='euclidean', std_noise=0., normalization=-1, device='cpu', autoregressive=False):
        super(PairDistanceAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.std_noise = std_noise
        self.distance = DISTANCE_TORCH[distance]
        self.autoregressive = autoregressive

        self.manifold = None
        if distance == 'hyperbolic':
            self.manifold = PoincareBall()
            print("Using poincare sampling")

        self.norm = normalization
        self.device = device

    def normalization(self, enc, hyp_max_norm=1-1e-2):
        if self.norm > 0:
            return self.norm * enc / torch.norm(enc, dim=-1, keepdim=True)
        elif self.manifold is not None:
            scaling = torch.clamp((hyp_max_norm / torch.norm(enc, dim=-1, keepdim=True)), max=1)
            return scaling * enc
        return enc

    def forward(self, sequence):
        (B, _, N) = sequence.shape
        sequence = sequence.reshape(2 * B, N)

        # encoder
        enc_sequence = self.encoder(sequence)

        # compute distance
        enc_dist = enc_sequence.reshape(B, 2, -1)
        distance = self.distance(enc_dist[:, 0], enc_dist[:, 1])

        # normalize if self.norm=True and add noise
        enc_sequence = self.normalization(enc_sequence)
        enc_sequence = self.sample(enc_sequence)
        enc_sequence = self.normalization(enc_sequence)

        # decode
        if self.autoregressive:
            dec_sequence = self.decoder(enc_sequence, sequence)
        else:
            dec_sequence = self.decoder(enc_sequence)
        dec_sequence = dec_sequence.reshape(B, 2, N, -1)
        return distance, dec_sequence

    def sample(self, mean):
        if self.manifold is None:
            return mean + self.std_noise * torch.randn_like(mean)
        else:
            # sampling Wrapped Normal distribution in the Poincarè Ball
            # for more details see https://arxiv.org/pdf/1901.06033.pdf
            with torch.no_grad():
                dim = mean.shape[-1]
                v = self.std_noise * torch.randn_like(mean)
                self.manifold.assert_check_vector_on_tangent(torch.zeros(1, dim).to(self.device), v)
                v = v / self.manifold.lambda_x(torch.zeros(1, dim).to(self.device), keepdim=True)
                u = self.manifold.transp(torch.zeros(1, dim).to(self.device), mean, v)
                z = self.manifold.expmap(mean, u)
                return z
