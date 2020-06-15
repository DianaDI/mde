from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import json

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
        im = ax.imshow(imgs[col]) #vmin=min, vmax=max)
        fig.colorbar(im, ax=ax)
    plt.savefig(f'{save_path}/{mode}_sample_{epoch}_{batch_idx}.png', dpi=300)
    plt.clf()
    plt.close('all')


def save_run_params(params, name):
    res = json.dumps(params)
    f = open(f"{name}.json", "w")
    f.write(res)
    f.close()
