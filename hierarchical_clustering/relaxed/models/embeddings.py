"""
    Part of this code was adapted from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

import torch
import torch.nn as nn

from hierarchical_clustering.relaxed.models.model import HypHCModel
from util.ml_and_math.poincare import project


class HypHCEmbeddings(HypHCModel):
    """ Hyperbolic embedding model for hierarchical clustering. """

    def __init__(self, n_nodes=1, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3):
        super(HypHCEmbeddings, self).__init__(temperature=temperature, init_size=init_size, max_scale=max_scale)
        self.n_nodes = n_nodes
        self.embeddings = nn.Embedding(n_nodes, rank)
        self.embeddings.weight.data = project(
            self.scale * (2 * torch.rand((n_nodes, rank)) - 1.0)
        )

    def encode(self, triple_ids, sequences=None):
        e = self.embeddings(triple_ids)
        return e
