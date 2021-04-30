import argparse
import pickle
import time
import numpy as np
import torch

from util.distance_functions.distance_matrix import DISTANCE_MATRIX
from closest_string.task.dataset import ReferenceDataset, QueryDataset
from util.ml_and_math.loss_functions import AverageMeter


def load_closest_match_dataset(path):
    with open(path, 'rb') as f:
        sequences_references, sequences_queries, labels = pickle.load(f)

    reference_dataset = ReferenceDataset(sequences_references)
    query_dataset = QueryDataset(sequences_queries, labels)
    return reference_dataset, query_dataset


def execute_test(args):
    """ Run when pretrained model is saved and needs to be restored """
    # set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'
    print('Using device:', device)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # model
    model_class, model_args, state_dict, distance = torch.load(args.encoder_path)
    encoder_model = model_class(**vars(model_args))

    # Restore best model
    print('Loading model ' + args.encoder_path)
    encoder_model.load_state_dict(state_dict)
    encoder_model.eval()

    closest_string_testing(encoder_model, args.data, args.batch_size, device, distance)


def closest_string_testing(encoder_model, data_path, batch_size, device, distance):
    """ Main routine: computes performance given pretrained model, dataset path and other arguments """
    t_total = time.time()

    # load data
    reference_dataset, query_dataset = load_closest_match_dataset(data_path)
    reference_loader = torch.utils.data.DataLoader(reference_dataset, batch_size=batch_size, shuffle=False)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    distance_function = DISTANCE_MATRIX[distance]

    # Testing
    embedded_reference = embed_strings(reference_loader, encoder_model, device)
    avg_acc = test(query_loader, encoder_model, embedded_reference, distance_function, device)
    print('Results: accuracy {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'
          .format(*avg_acc))
    print('Top1: {:.3f}  Top5: {:.3f}  Top10: {:.3f}'.format(avg_acc[0], avg_acc[4], avg_acc[9]))

    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))


def embed_strings(loader, model, device):
    """ Embeds the sequences of a dataset one batch at the time given an encoder """
    embedded_list = []

    for sequences in loader:
        sequences = sequences.to(device)
        embedded = model.encode(sequences)
        embedded_list.append(embedded.cpu().detach())

    embedded_reference = torch.cat(embedded_list, axis=0)
    return embedded_reference


def test(loader, model, embedded_reference, distance, device, slots=10):
    """ Given the embedding of the references, embeds and checks the performance for one batch of queries at a time """
    avg_acc = AverageMeter(len_tuple=slots)
    embedded_reference = embedded_reference.to(device)

    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        embedded = model.encode(sequences)

        distance_matrix = distance(embedded_reference, embedded, model.scaling)

        label_distances = distance_matrix[labels.long(), torch.arange(0, distance_matrix.shape[1])]
        rank = torch.sum(torch.le(distance_matrix, label_distances.unsqueeze(0)).float(), dim=0)

        acc = [torch.mean((rank <= i + 1).float()) for i in range(slots)]
        avg_acc.update(acc, sequences.shape[0])

    return avg_acc.avg


if __name__ == '__main__':
    """ Run when pretrained model is saved and needs to be restored """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/closest_small.pkl', help='Dataset path')
    parser.add_argument('--encoder_path', type=str, default='./data/CNN.pkl', help='Pretrained model path')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA (GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()

    execute_test(args)
