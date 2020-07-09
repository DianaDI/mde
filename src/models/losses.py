import torch
import torch.nn as nn


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean(torch.abs(grad_real - grad_fake))


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))
        return 1 - torch.mean(prod / (fake_norm * real_norm))


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    # L1 norm
    def forward(self, pred, target, mask, device, factor=0.6):
        # assuming mask consists of 0, 1 values
        mask = (mask * factor).to(device, dtype=torch.float)
        ones = (torch.ones(mask.shape).to(device) * (1 - factor)).to(device)
        mask = torch.where(mask == 0, ones, mask).to(device, dtype=torch.float)
        return torch.mean(torch.abs(target - pred * mask))
