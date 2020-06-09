from matplotlib import pyplot as plt
import matplotlib
from src.models import NORMALIZE
from src.data.transforms import minmax, minmax_over_nonzero
import numpy as np

matplotlib.use('Agg')


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
    # normalize image between its own bounds for better visualisation
    output = minmax_over_nonzero(output)
    target = minmax_over_nonzero(target)
    mask = (target >= 0).astype(int)
    output = np.multiply(output, mask)
    target = np.multiply(target, mask)
    im1 = axes.flat[0].imshow(output)
    im2 = axes.flat[1].imshow(target)
    fig.colorbar(im1, ax=axes.ravel().tolist())
    fig.colorbar(im2, ax=axes.ravel().tolist())
    plt.savefig(f'{save_path}/{mode}_sample_{epoch}_{batch_idx}.png', dpi=300)
    plt.clf()
    plt.close('all')
