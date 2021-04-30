import torch
from util.data_handling.data_loader import index_to_one_hot


class ReferenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = index_to_one_hot(sequences)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, index):
        return self.sequences[index]


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = index_to_one_hot(sequences)
        self.labels = labels

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]