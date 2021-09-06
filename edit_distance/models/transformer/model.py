import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.ml_and_math.layers import MLP


class Transformer(nn.Module):

    def __init__(self, len_sequence, segment_size, embedding_size, hidden_size, trans_layers, readout_layers, device, alphabet_size=4,
                 dropout=0.0, heads=1, layer_norm=False, mask='empty'):
        super(Transformer, self).__init__()

        self.segment_size = segment_size
        self.padding = (- len_sequence) % segment_size
        len_sequence += self.padding
        print("padding", self.padding)

        if mask == "empty":
            self.mask_sequence = generate_empty_mask(len_sequence//segment_size).to(device)
        elif mask == "no_prev":
            self.mask_sequence = generate_square_previous_mask(len_sequence//segment_size).to(device)
        elif mask[:5] == "local":
            self.mask_sequence = generate_local_mask(len_sequence//segment_size, k=int(mask[5:])).to(device)

        self.sequence_trans = TransformerEncoderModel(ntoken=alphabet_size*segment_size, nout=hidden_size, ninp=hidden_size,
                                                  nhead=heads, nhid=hidden_size, nlayers=trans_layers, dropout=dropout,
                                                  layer_norm=layer_norm, max_len=len_sequence//segment_size)

        self.readout = MLP(in_size=len_sequence // segment_size * hidden_size, hidden_size=embedding_size, out_size=embedding_size, layers=readout_layers,
                           dropout=dropout, device=device)

        self.to(device)

    def forward(self, sequence):

        # initial padding
        if self.padding > 0:
            sequence = F.pad(sequence, (0,0,0,self.padding))

        # sequence (B, N, 4)
        (B, N, _) = sequence.shape

        # apply attention layers
        sequence = sequence.reshape((B, N//self.segment_size, -1)).transpose(0, 1)
        enc_sequence = self.sequence_trans(sequence, self.mask_sequence)

        # apply readout
        enc_sequence = enc_sequence.transpose(0, 1).reshape(B, -1)
        embedding = self.readout(enc_sequence)
        return embedding


class TransformerEncoderModel(nn.Module):
    """ Part of this code was adapted from the examples of the PyTorch library """

    def __init__(self, ntoken, nout, ninp, nhead, nhid, nlayers, max_len, dropout=0.0, layer_norm=False):
        super(TransformerEncoderModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, norm= \
            nn.LayerNorm(normalized_shape=ninp, eps=1e-6) if layer_norm else None)
        self.encoder = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_previous_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_local_mask(sz, k=3):
    mask = torch.eye(sz)
    for i in range(1, k + 1):
        mask += torch.cat((torch.zeros(i, sz), torch.eye(sz)[:-i]), dim=0)
        mask += torch.cat((torch.zeros(sz, i), torch.eye(sz)[:, :-i]), dim=1)

    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_empty_mask(sz):
    mask = torch.zeros(sz, sz)
    return mask