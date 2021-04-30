import Levenshtein
import torch
import torch.nn.functional as F

from util.ml_and_math.loss_functions import MAPE, AverageMeter


def pair_autoencoder_loss(decoded_sequences, predicted_distance, input_sequences, distance_label, alpha):
    pred = torch.argmax(decoded_sequences, dim=-1)
    reconstruction_accuracy = torch.mean(torch.eq(pred, input_sequences).float())
    decoded_sequences = decoded_sequences.permute(0, 3, 1, 2)  # (B, 2, N, 4) -> (B, 4, 2, N)

    distance_loss = F.mse_loss(predicted_distance, distance_label)
    distance_mape = MAPE(predicted_distance, distance_label)

    reconstruction_loss = F.cross_entropy(decoded_sequences, input_sequences)
    loss = (1 - alpha) * distance_loss + alpha * reconstruction_loss
    return loss, (distance_loss.data.item(), distance_mape.data.item(),
                  reconstruction_loss.data.item(), reconstruction_accuracy.data.item())


def multiple_alignment_cost(sequences, centers, distance=Levenshtein.distance, alphabet_size=4):
    (B, K, N) = sequences.shape
    centers = torch_to_string(centers, alphabet_size)

    mean_distance = AverageMeter()

    for i in range(B):
        strings = torch_to_string(sequences[i], alphabet_size)

        for j in range(K):
            d = distance(strings[j], centers[i])
            mean_distance.update(d)

    return mean_distance.avg


def remove_padding(s, alphabet_size=4):
    while s[-1] == alphabet_size:
        s = s[:-1]
    return s


def torch_to_string(S, alphabet_size=4):
    S = S.tolist()
    S = [remove_padding(s, alphabet_size) for s in S]
    S = [[str(l) for l in s] for s in S]
    S = [''.join(s) for s in S]
    return S


def torch_to_string2(sequences, alphabet_size=4):
    sequences = sequences.tolist()
    strings = []
    for S in sequences:
        S = [remove_padding(s, alphabet_size) for s in S]
        S = [[str(l) for l in s] for s in S]
        S = [''.join(s) for s in S]
        strings.append(S)
    return strings



