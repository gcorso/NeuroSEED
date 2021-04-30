"""
    Part of this code was adapted from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from hierarchical_clustering.relaxed.utils.lca import hyp_lca
from hierarchical_clustering.relaxed.utils.linkage import nn_merge_uf_fast_np, sl_from_embeddings
from util.ml_and_math.poincare import project


def index_to_one_hot(x, alphabet_size=4, device='cpu'):
    # add one row of zeros because the -1 represents the absence of element and it is encoded with zeros
    x = torch.cat((torch.eye(alphabet_size, device=device), torch.zeros((1, alphabet_size), device=device)), dim=0)[x]
    return x


class HypHCModel(nn.Module):
    """ Class interface for the other hyperbolic models for hierarchical clustering """

    def __init__(self, temperature=0.05, init_size=1e-2, max_scale=1. - 1e-3):
        super(HypHCModel, self).__init__()
        self.temperature = temperature
        self.scale = nn.Parameter(torch.Tensor([init_size]), requires_grad=True)
        self.init_size = init_size
        self.max_scale = max_scale

    def anneal_temperature(self, anneal_factor):
        """ anneal_factor: scalar for temperature decay """
        self.temperature *= anneal_factor

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=-1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)

    def encode(self, triple_ids, sequences):
        pass

    def loss(self, triple_ids, sequences, similarities):
        """ Computes the HypHCEmbeddings loss.
        Args:
            triple_ids: B x 3 tensor with triple ids
            sequences:  B x 3 x N tensor with elements indexes
            similarities: B x 3 tensor with pairwise similarities for triples 
                          [s12, s13, s23]
        """
        (B, _, N) = sequences.shape
        triple_ids, sequences = triple_ids.reshape((3*B)), sequences.reshape((3*B, N))

        e = self.encode(triple_ids, sequences)
        e = self.normalize_embeddings(e)
        e = e.reshape((B, 3, -1))

        d_12 = hyp_lca(e[:, 0], e[:, 1], return_coord=False)
        d_13 = hyp_lca(e[:, 0], e[:, 2], return_coord=False)
        d_23 = hyp_lca(e[:, 1], e[:, 2], return_coord=False)
        lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
        weights = torch.softmax(lca_norm / self.temperature, dim=-1)
        w_ord = torch.sum(similarities * weights, dim=-1, keepdim=True)
        total = torch.sum(similarities, dim=-1, keepdim=True) - w_ord
        return torch.mean(total)

    def decode_tree(self, embedded_sequences, fast_decoding):
        """Build a binary tree (nx graph) from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self.normalize_embeddings(embedded_sequences)
        leaves_embeddings = project(leaves_embeddings).detach().cpu()
        sim_fn = lambda x, y: torch.matmul(x, y.transpose(0, 1))
        if fast_decoding:
            parents = nn_merge_uf_fast_np(leaves_embeddings, S=sim_fn, partition_ratio=1.2)
        else:
            parents = sl_from_embeddings(leaves_embeddings, sim_fn)
        tree = nx.DiGraph()
        for i, j in enumerate(parents[:-1]):
            tree.add_edge(j, i)
        return tree
