from matplotlib import pyplot as plt
import matplotlib

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
    im = axes.flat[0].imshow(output)
    im = axes.flat[1].imshow(target)
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(f'{save_path}/{mode}_sample_{epoch}_{batch_idx}.png', dpi=300)
    plt.clf()
    plt.close('all')
