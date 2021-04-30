import torch

class AverageMeter(object):
    def __init__(self, len_tuple=0):
        self.len_tuple = len_tuple
        self.avg = 0 if len_tuple == 0 else tuple([0] * len_tuple)
        self.sum = 0 if len_tuple == 0 else [0] * len_tuple
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        if self.len_tuple == 0:
            self.sum += (val.data.item() if torch.is_tensor(val) else val) * n
            self.avg = self.sum / self.count
        else:
            self.sum = [self.sum[i] + (val[i].data.item() if torch.is_tensor(val) else val[i]) * n for i in range(self.len_tuple)]
            self.avg = tuple(s / self.count for s in self.sum)


def accuracy(output, target):
    estimate = torch.argmax(output, dim=1)
    acc = torch.mean(torch.eq(estimate, target).float())
    return acc.item()


def MAPE(pred, label, eps=1e-6):
    return torch.mean(torch.abs(pred - label) / (label + eps))
