import pickle
import sys

import torch


def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_dataloaders(datasets, batch_size, workers):
    loaders = {}
    for key in datasets.keys():
        loaders[key] = torch.utils.data.DataLoader(datasets[key], batch_size=batch_size,
                                                   shuffle=True, num_workers=workers)
    return loaders


def index_to_one_hot(x, alphabet_size=4, device='cpu'):
    # add one row of zeros because the -1 represents the absence of element and it is encoded with zeros
    x = torch.cat((torch.eye(alphabet_size, device=device), torch.zeros((1, alphabet_size), device=device)), dim=0)[x]
    return x


def dataset_to_one_hot(dic, alphabet_size=4):
    for dset in dic.keys():
        dic[dset] = index_to_one_hot(dic[dset], alphabet_size=alphabet_size)


BOOL_CHOICE = ['True', 'False']