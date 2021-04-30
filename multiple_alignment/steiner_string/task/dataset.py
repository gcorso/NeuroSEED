import random
import torch


class PairDistanceDatasetSampled(torch.utils.data.Dataset):
    def __init__(self, sequences, distances, multiplicity=1):
        self.len_sequence = sequences.shape[-1]
        sequences[sequences == -1] = 4
        self.sequences = sequences
        self.distances = distances
        self.N_batches = self.sequences.shape[0]
        self.batch_size = self.sequences.shape[1]
        self.multiplicity = multiplicity

        # Normalise labels
        self.normalisation_constant = self.sequences.shape[-1]
        self.distances = self.distances / self.normalisation_constant

        print("Size dataset ", self.__len__())

    def __len__(self):
        return self.N_batches * self.batch_size * self.multiplicity

    def __getitem__(self, index):
        index = index // self.multiplicity
        sequences = None
        d = 0
        while d == 0:
            idx2 = random.randint(0, self.batch_size-1)
            sequences = [self.sequences[index // self.batch_size, index % self.batch_size].unsqueeze(0),
                     self.sequences[index // self.batch_size, idx2].unsqueeze(0)]
            sequences = torch.cat(sequences, dim=0)
            d = self.distances[index // self.batch_size, index % self.batch_size, idx2]
        return sequences, d


class PairDistanceDatasetFull(torch.utils.data.Dataset):
    def __init__(self, sequences, distances):
        self.len_sequence = sequences.shape[-1]

        sequences[sequences == -1] = 4

        self.sequences = sequences
        self.distances = distances
        self.N_sequences = sequences.shape[0]

        # Normalise labels
        self.normalisation_constant = self.sequences.shape[-1]
        self.distances = self.distances / self.normalisation_constant

        print("Size dataset ", self.__len__())

    def __len__(self):
        return self.N_sequences * (self.N_sequences - 1)

    def __getitem__(self, index):
        idx1 = index // (self.N_sequences - 1)
        idx2 = index % (self.N_sequences - 1)
        if idx2 >= idx1:
            idx2 += 1

        sequences = [self.sequences[idx1].unsqueeze(0), self.sequences[idx2].unsqueeze(0)]
        sequences = torch.cat(sequences, dim=0)
        d = self.distances[idx1, idx2]
        return sequences, d


class MultipleAlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.len_sequence = sequences.shape[-1]
        sequences[sequences == -1] = 4
        self.sequences = sequences

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, index):
        return self.sequences[index]
