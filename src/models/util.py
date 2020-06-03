from matplotlib import pyplot as plt
import numpy as np


def plot_metrics(metrics, names, save_path):
    if len(metrics) != len(names):
        print("Metric or metric name is missing! Cannot plot.")
    else:
        for metric, name in metrics, names:
            plt.plot(metric)
            plt.ylabel(name)
            plt.savefig(f'{save_path}/{name}.png', dpi=300)


def plot_sample(output, target, save_path, batch_idx, mode):
    img = np.hstack((output, target))
    plt.imsave(f'{save_path}/{mode}_sample_{batch_idx}.png', img, dpi=300)
