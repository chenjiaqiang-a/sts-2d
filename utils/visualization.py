import numpy as np
import matplotlib
import matplotlib.pyplot as plt

__all__ = ['plot_mean_std_curves']


def plot_mean_std_curves(curves: dict, strids: dict, labels: dict, filename):
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 18,
        'mathtext.fontset': 'stix',
        'font.sans-serif': ['SimHei'],
    })
    plt.figure(figsize=(7, 5), dpi=100)

    def mean_std_list(value_list, strid):
        values = np.array(value_list)
        m = np.mean(values, axis=0)
        s = np.std(values, axis=0)
        index = np.arange(strid, len(m) * strid + 1, strid)
        return index, m, s

    for i, key in enumerate(curves.keys()):
        ids, mean, std = mean_std_list(curves[key], strids[key])
        plt.plot(ids, mean, color=f'C{i}', label=labels[key])
        plt.fill_between(ids, mean - std, mean + std, color=f'C{i}', alpha=0.3)

    plt.legend()
    plt.savefig(filename)
    plt.show()
