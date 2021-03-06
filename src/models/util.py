from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import numpy as np
import torch
import torch.nn as nn
import cv2

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

    # uncomment to plot separate samples for report, comment section below
    # plt.gca().set_axis_off()
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.margins(0, 0)
    # plt.imshow(imgs[2])
    # plt.savefig(f'{save_path}/{mode}_output_single_{batch_idx}.png', dpi=150, bbox_inches = 'tight',
    # pad_inches = 0)
    #
    # plt.imshow(imgs[1])
    # plt.savefig(f'{save_path}/{mode}_target_single_{batch_idx}.png', dpi=150, bbox_inches = 'tight',
    # pad_inches = 0)
    #
    # plt.imshow(imgs[0])
    # plt.savefig(f'{save_path}/{mode}_input_single_{batch_idx}.png', dpi=150, bbox_inches = 'tight',
    # pad_inches = 0)

    # plot samples in a grid
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
    if mode != "eval":
        ax3.title.set_text('Edges')
        im3 = ax3.imshow(imgs[3], vmin=np.min(imgs[3]), vmax=np.max(imgs[3]))
    else:
        ax3.title.set_text('L1-loss')
        im3 = ax3.imshow(imgs[4], vmin=np.min(imgs[4]), vmax=np.max(imgs[4]))
        cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax3)

    if mode != "eval":
        ax4 = axes[1][2]
        ax4.title.set_text('Weighted pixelwise L1-loss')
        im4 = ax4.imshow(imgs[4], vmin=np.min(imgs[4]), vmax=np.max(imgs[4]))
        cax4 = make_axes_locatable(ax4).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax4)
    else:
        fig.delaxes(axes[1][2])

    fig.delaxes(axes[0][2])
    fig.tight_layout(pad=1.0)
    plt.savefig(f'{save_path}/{mode}_sample_{epoch}_{batch_idx}.png', dpi=300)
    plt.clf()
    plt.close('all')


def log_sample(cur_batch, plot_every, out, target, inp, edges, pixel_loss, path, epoch, mode):
    if cur_batch % plot_every == 0:
        plot_sample(cv2.merge((inp[0][0, :, :].numpy(),
                               inp[0][1, :, :].numpy(),
                               inp[0][2, :, :].numpy())),
                    out[0][0, :, :].cpu().detach().numpy(),
                    target[0][0, :, :].cpu().detach().numpy(),
                    edges[0][0, :, :].cpu().numpy() if mode != "eval" else None,
                    pixel_loss[0][0, :, :].cpu().detach().numpy(),
                    path, epoch, cur_batch, mode)


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
