from matplotlib import pyplot as plt

def plot_metrics(metrics, names, save_path):
    if len(metrics) != len(names):
        print("Metric or metric name is missing! Cannot plot.")
    else:
        for metric, name in metrics, names:
            plt.plot(metric)
            plt.ylabel(name)
            plt.savefig(f'{save_path}/{name}.png', dpi=300)


def plot_sample(output, target, save_path, batch_idx, mode):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Predicted VS Ground truth')
    ax1.imshow(output)
    ax2.imshow(target)
    plt.savefig(f'{save_path}/{mode}_sample_{batch_idx}.png', dpi=300)