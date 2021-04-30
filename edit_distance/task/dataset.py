import random
import torch
from util.data_handling.data_loader import index_to_one_hot


class EditDistanceDatasetSampled(torch.utils.data.Dataset):
    # only some pairwise distances available are randomly sampled at every epoch

    def __init__(self, sequences, distances, multiplicity=1):
        # multiplicity indicates (1/2) the number of distances sampled per sequence at every epoch

        self.len_sequence = sequences.shape[-1]
        self.sequences = index_to_one_hot(sequences)
        self.distances = distances
        self.N_batches = self.sequences.shape[0]
        self.batch_size = self.sequences.shape[1]
        self.multiplicity = multiplicity

        # Normalise labels
        self.normalisation_constant = self.sequences.shape[-2]
        self.distances = self.distances / self.normalisation_constant

    def __len__(self):
        return self.N_batches * self.batch_size * self.multiplicity

    def __getitem__(self, index):
        index = index // self.multiplicity
        sequences = None
        d = torch.Tensor([0.0])
        while torch.all(d == 0):  # avoid equal sequences that might give numerical problems
            idx2 = random.randint(0, self.batch_size-1)
            sequences = [self.sequences[index // self.batch_size, index % self.batch_size].unsqueeze(0),
                     self.sequences[index // self.batch_size, idx2].unsqueeze(0)]
            sequences = torch.cat(sequences, dim=0)
            d = self.distances[index // self.batch_size, index % self.batch_size, idx2]
        return sequences, d


class EditDistanceDatasetComplete(torch.utils.data.Dataset):
    # every pairwise distance is given loaded at every epoch

    def __init__(self, sequences, distances):
        self.len_sequence = sequences.shape[-1]
        self.sequences = index_to_one_hot(sequences)
        self.distances = distances
        self.N_sequences = sequences.shape[0]

        # Normalise labels
        self.normalisation_constant = self.sequences.shape[-2]
        self.distances = self.distances / self.normalisation_constant

    def __len__(self):
        return self.N_sequences * (self.N_sequences - 1)

    def __getitem__(self, index):
        # calculate the right indices avoiding pairs (i, i)
        idx1 = index // (self.N_sequences -1)
        idx2 = index % (self.N_sequences-1)
        if idx2 >= idx1:
            idx2+=1

        sequences = [self.sequences[idx1].unsqueeze(0), self.sequences[idx2].unsqueeze(0)]
        sequences = torch.cat(sequences, dim=0)
        d = self.distances[idx1, idx2]
        return sequences, d


class EditDistanceDatasetSupervision(torch.utils.data.Dataset):
    def __init__(self, sequences, distances):
        self.len_sequence = sequences.shape[-1]

        self.sequences = index_to_one_hot(sequences)
        self.distances = distances
        self.N_batches = self.sequences.shape[0]
        self.batch_size = self.sequences.shape[1]

        # Normalise labels
        self.normalisation_constant = self.sequences.shape[-2]
        self.distances = [d / (self.normalisation_constant * 2**p) for p, d in enumerate(self.distances)]

    def __len__(self):
        return self.N_batches * self.batch_size

    def __getitem__(self, index):
        sequences = None
        ds = [0]
        while ds[0] == 0:
            idx2 = random.randint(0, self.batch_size-1)
            sequences = [self.sequences[index // self.batch_size, index % self.batch_size].unsqueeze(0),
                     self.sequences[index // self.batch_size, idx2].unsqueeze(0)]
            sequences = torch.cat(sequences, dim=0)
            ds = [d[index // self.batch_size, index % self.batch_size, idx2] for d in self.distances]
        return sequences, ds