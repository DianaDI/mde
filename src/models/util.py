from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import numpy as np
import torch
import torch.nn as nn

matplotlib.use('Agg')
plt.rcParams.update({'font.size': 7})


def plot_metrics(metrics, names, save_path, mode):
    if len(metrics) != len(names):
        print("Metric or metric name is missing! Cannot plot.")
    else:
        for metric, name in zip(metrics, names):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(metric)
            plt.title(name)
            plt.savefig(f'{save_path}/{mode}_{name}.png', dpi=300)
            plt.clf()
            plt.close('all')


def plot_sample(orig, output, target, edges, pixel_loss, save_path, epoch, batch_idx, mode):
    imgs = [orig, target, output, edges, pixel_loss]
    fig, axes = plt.subplots(nrows=2, ncols=3)

    ax0 = axes[0][0]
    ax0.title.set_text('Original img')
    im0 = ax0.imshow(imgs[0])

    ax1 = axes[1][0]
    ax1.title.set_text('Target DM')
    im1 = ax1.imshow(imgs[1], vmin=np.min(imgs[1]), vmax=np.max(imgs[1]))
    cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)

    ax2 = axes[1][1]
    ax2.title.set_text('Predicted DM')
    im2 = ax2.imshow(imgs[2], vmin=np.min(imgs[2]), vmax=np.max(imgs[2]))
    cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2)

    ax3 = axes[0][1]
    ax3.title.set_text('Edges')
    im3 = ax3.imshow(imgs[3], vmin=np.min(imgs[3]), vmax=np.max(imgs[3]))

    ax4 = axes[1][2]
    if mode == "eval":
        ax4.title.set_text('L1-loss')
    else:
        ax4.title.set_text('Weighted pixelwise L1-loss')
    im4 = ax4.imshow(imgs[4], vmin=np.min(imgs[4]), vmax=np.max(imgs[4]))
    cax4 = make_axes_locatable(ax4).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im4, cax=cax4)

    fig.delaxes(axes[0][2])

    fig.tight_layout(pad=1.0)
    plt.savefig(f'{save_path}/{mode}_sample_{epoch}_{batch_idx}.png', dpi=300)
    plt.clf()
    plt.close('all')


def save_dm(output, target, save_path, batch_idx, mode="eval"):
    # output, target are supposed to be numpy arrays
    output.dump(f'{save_path}/{mode}_dm_output_{batch_idx}.dmp')
    target.dump(f'{save_path}/{mode}_dm_target_{batch_idx}.dmp')


def save_dict(params, name):
    res = json.dumps(params)
    f = open(f"{name}.json", "w")
    f.write(res)
    f.close()


def imgrad(img, device):
    img = torch.mean(img, 1, True)
    # grad x
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0).to(device)
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0).to(device)
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x


def imgrad_yx(img, device):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img, device)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)
