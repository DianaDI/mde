import numpy as np
import torch
from torch.autograd import Variable
from src.models.metrics import compute_metrics
from src.models.util import log_sample, save_dm, save_dict, imgrad_yx


def get_prediction(data, model, device, interpolate):
    orig_inp = data['image'].permute(0, 3, 1, 2)
    inp = Variable(orig_inp).to(device, dtype=torch.float)
    target = Variable(data['depth']).to(device, dtype=torch.float).unsqueeze(1)
    mask = data['mask'].to(device).unsqueeze(1)
    edges = Variable(data['edges']).to(device).unsqueeze(1)
    range_min = data['range_min'].to(device, dtype=torch.float)
    range_max = data['range_max'].to(device, dtype=torch.float)
    out = model(inp)
    if not interpolate:
        out = out * mask
        target = target * mask
    return inp, out, target, edges.detach(), orig_inp.detach(), range_min.detach(), range_max.detach()


def calc_loss(data, model, l1_criterion, criterion_img, criterion_norm, device, interpolate, edge_factor, batch_idx, reg=False):
    inp, out, target, edges, orig_inp, _, _ = get_prediction(data, model, device, interpolate)
    imgrad_true = imgrad_yx(target, device)
    imgrad_out = imgrad_yx(out, device)
    l1_loss, l1_losses = l1_criterion(out, target, edges, device, factor=edge_factor)
    loss_grad = criterion_img(imgrad_out, imgrad_true)
    loss_normal = criterion_norm(imgrad_out, imgrad_true)
    total_loss = l1_loss + 0.5 * loss_grad + 0.5 * loss_normal  # + 0.5 * loss_ssim
    if reg:
        loss_reg = Variable(torch.tensor(0.)).to(device)
        for param in model.parameters():
            loss_reg = loss_reg + param.norm(2)
        total_loss = total_loss + 1e-20 * loss_reg
    if batch_idx % 10 == 0:
        print(f'DM l1-loss: {l1_loss.item()}, '
              f'Loss Grad: {loss_grad.item()},  '
              f'Loss Normal: {loss_normal.item()}, ')
    return total_loss, l1_losses, out, target, inp, orig_inp, edges


def eval(model, test_loader, device, interpolate, model_save_path, results_save_path, fig_save_path, plot=True):
    print("\nEVALUATION STARTING...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    rmse_abs, rmse, l1, l1_abs = [], [], [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            _, out, target, _, orig_inp, range_min, range_max = get_prediction(data, model, device, interpolate)
            rmse_loss, l1_loss, pixel_losses, l1_abs_loss, rmse_abs_loss = compute_metrics(out, target, range_min, range_max)
            rmse.append(rmse_loss)
            l1.append(l1_loss)
            rmse_abs.append(rmse_abs_loss)
            l1_abs.append(l1_abs_loss)
            if plot:
                log_sample(batch_idx, 50, out, target, orig_inp, None, pixel_losses, fig_save_path, "", "eval")
                if batch_idx % 100 == 0:
                    save_dm(out[0][0, :, :].cpu().detach().numpy(), target[0][0, :, :].cpu().detach().numpy(), fig_save_path, batch_idx)
    results = {
        "Mean RMSE Loss": np.mean(rmse),
        "Mean L1 Loss": np.mean(l1),
        "Mean RMSE Abs Loss": np.mean(rmse_abs),
        "Mean L1 Abs Loss": np.mean(l1_abs)
    }
    for key in results:
        print(f'{key}: {results[key]}')
    save_dict(results, results_save_path)
