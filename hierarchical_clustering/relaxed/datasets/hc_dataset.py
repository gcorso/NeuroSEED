"""
    Part of this code was adapted from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

import logging
import numpy as np
import torch
import torch.utils.data as data
import pickle

from hierarchical_clustering.relaxed.datasets.triples import generate_all_triples, samples_triples


def load_hc_data(dataset):
    with open(dataset, 'rb') as f:
        x, similarities = pickle.load(f)
    return x, similarities


class HCTripletDataset(data.Dataset):
    """Hierarchical clustering dataset."""

    def __init__(self, sequences, similarities, num_samples):
        """Creates Hierarchical Clustering dataset with triples. """
        self.sequences = sequences
        self.similarities = similarities
        self.n_nodes = self.similarities.shape[0]
        self.triples = self.generate_triples(num_samples)
        self.len_sequences = sequences.shape[-1]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        seq1 = self.sequences[triple[0]]
        seq2 = self.sequences[triple[1]]
        seq3 = self.sequences[triple[2]]
        sequences = np.array([seq1, seq2, seq3])
        s12 = self.similarities[triple[0], triple[1]]
        s13 = self.similarities[triple[0], triple[2]]
        s23 = self.similarities[triple[1], triple[2]]
        similarities = np.array([s12, s13, s23])
        return torch.from_numpy(triple), torch.from_numpy(sequences),torch.from_numpy(similarities)

    def generate_triples(self, num_samples):
        logging.info("Generating triples.")
        if num_samples < 0:
            triples = generate_all_triples(self.n_nodes)
        else:
            triples = samples_triples(self.n_nodes, num_samples=num_samples)
        logging.info("Total of {} triples".format(triples.shape[0]))
        return triples.astype("int64")
