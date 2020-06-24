from matplotlib import pyplot as plt
import matplotlib
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


def plot_sample(output, target, save_path, epoch, batch_idx, mode):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    imgs = [output, target]
    for col in range(2):
        ax = axes[col]
        # min = np.min(imgs[col][np.nonzero(imgs[col])])
        # max = np.max(imgs[col])
        im = ax.imshow(imgs[col])  # vmin=min, vmax=max)
        fig.colorbar(im, ax=ax)
    plt.savefig(f'{save_path}/{mode}_sample_{epoch}_{batch_idx}.png', dpi=300)
    plt.clf()
    plt.close('all')


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
