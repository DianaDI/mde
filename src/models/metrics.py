import torch


def mean_relative_error(output, target):
    # todo might not be valid with normalization
    target_no_zeros = torch.where(target > 0, target, target + 0.001)
    loss = torch.mean(torch.abs(output - target) / torch.abs(target_no_zeros))
    return loss.item()


def root_mean_squared_error(output, target):
    loss = torch.sqrt(torch.mean((output - target) ** 2))
    return loss.item()
