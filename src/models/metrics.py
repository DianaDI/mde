import torch
from src.data.transforms import minmax_reverse

def mean_relative_error(output, target):
    # todo might not be valid with normalization
    target_no_zeros = torch.where(target > 0, target, target + 0.001)
    loss = torch.mean(torch.abs(output - target) / torch.abs(target_no_zeros))
    return loss.item()


def root_mean_squared_error(output, target):
    loss = torch.mean(torch.sqrt((output - target) ** 2))
    return loss.item()


def get_absolute_labels(output, target, min, max):
    for batch in range(len(output)):
        output[batch] = minmax_reverse(output[batch], min[batch], max[batch])
    for batch in range(len(target)):
        target[batch] = minmax_reverse(target[batch], min[batch], max[batch])
    return output, target

def l1_absolute_error(output, target, min, max):
    output, target = get_absolute_labels(output, target, min, max)
    return torch.mean(torch.abs(target - output)).item()


def rmse_absolute(output, target, min, max):
    output, target = get_absolute_labels(output, target, min, max)
    return root_mean_squared_error(output, target)