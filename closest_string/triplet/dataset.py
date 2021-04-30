import random
import torch
from util.data_handling.data_loader import index_to_one_hot


class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, sequences, distances, multiplicity=1):
        # multiplicity indicates (1/2) the number of times a string is sampled at every epoch

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
        idx2, idx3 = 0, 0
        zero, d2, d3 = torch.Tensor([0.0]), torch.Tensor([0.0]), torch.Tensor([0.0])

        while torch.equal(d2, zero) or torch.equal(d3, zero) or torch.equal(d2, d3):
            idx2 = random.randint(0, self.batch_size - 1)
            idx3 = random.randint(0, self.batch_size - 1)
            d2 = self.distances[index // self.batch_size, index % self.batch_size, idx2]
            d3 = self.distances[index // self.batch_size, index % self.batch_size, idx3]

        if d2 < d3:
            sequences = [self.sequences[index // self.batch_size, index % self.batch_size].unsqueeze(0),
                         self.sequences[index // self.batch_size, idx2].unsqueeze(0),
                         self.sequences[index // self.batch_size, idx3].unsqueeze(0)]
        else:
            sequences = [self.sequences[index // self.batch_size, index % self.batch_size].unsqueeze(0),
                         self.sequences[index // self.batch_size, idx3].unsqueeze(0),
                         self.sequences[index // self.batch_size, idx2].unsqueeze(0)]
        sequences = torch.cat(sequences, dim=0)

        return sequences

