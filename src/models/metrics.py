import torch
import math
from src.data.transforms import minmax_reverse


def mean_relative_error(output, target):
    # not be valid with normalization
    target_no_zeros = torch.where(target > 0, target, target + 0.001)
    loss = torch.mean(torch.abs(output - target) / torch.abs(target_no_zeros))
    return loss.item()


def root_mean_squared_error(output, target):
    # loss = torch.mean(torch.sqrt((output - target) ** 2))
    loss = torch.nn.MSELoss().forward(output, target)
    return math.sqrt(loss.item())


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


def compute_metrics(output, target, min, max):
    rmse = root_mean_squared_error(output, target)
    l1, pixel_losses = torch.nn.L1Loss().forward(output, target)
    l1_abs = l1_absolute_error(output, target, min, max)
    rmse_abs = rmse_absolute(output, target, min, max)
    print(f'RMSE: {rmse}, L1: {l1}, RMSE abs: {rmse_abs}, L1 abs: {l1_abs}')
    return rmse, l1.item(), pixel_losses, l1_abs, rmse_abs
