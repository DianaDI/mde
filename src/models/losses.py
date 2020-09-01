import torch
import torch.nn as nn


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_fake, grad_real):
        return nn.L1Loss()(grad_real, grad_fake)


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))
        return 1 - torch.mean(prod / ((fake_norm * real_norm) + 1e-10))


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    # L1 norm
    def forward(self, pred, target, mask, device, factor=0.6):
        # assuming mask consists of 0, 1 values
        mask = (mask * factor).to(device, dtype=torch.float)
        ones = (torch.ones(mask.shape).to(device) * (1 - factor)).to(device)
        mask = torch.where(mask == 0, ones, mask).to(device, dtype=torch.float)
        losses = torch.abs((target - pred) * mask)
        return torch.mean(losses), losses


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    # L1 norm
    def forward(self, pred, target):
        losses = torch.abs(target - pred)
        return torch.mean(losses), losses


class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, pred, target):
        losses = nn.SmoothL1Loss(reduction="none")(pred, target)
        return torch.mean(losses), losses


class MaskedHuberLoss(nn.Module):
    def __init__(self):
        super(MaskedHuberLoss, self).__init__()

    def forward(self, pred, target, mask, device, factor=0.6):
        mask = (mask * factor).to(device, dtype=torch.float)
        ones = (torch.ones(mask.shape).to(device) * (1 - factor)).to(device)
        mask = torch.where(mask == 0, ones, mask).to(device, dtype=torch.float)
        losses = nn.SmoothL1Loss(reduction="none")(pred, target)
        masked = losses*mask
        return torch.mean(masked), masked