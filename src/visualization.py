import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import record_file

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = []
for k in [3,5, 7, 9]:
    data.append(np.array(pd.read_csv(record_file(k), header=None)))
data = np.array(data)


def show_fig(data_arr, legends, xlabel, ylabel, title, pos):
    plt.subplot(pos[0], pos[1], pos[2])
    for d in data_arr:
        plt.plot(range(d.shape[0]), d)

    plt.legend(legends)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def show_loss():
    d = [data[0, :, 1], data[1, :, 1], data[2, :, 1], data[3, :, 1]]
    lgd = ['ks=3', 'ks=5', 'ks=7', 'ks=9']
    show_fig(d, lgd, 'Epoch', 'Loss', '训练损失', (2, 2, 1))

    d = [data[0, :, 3], data[1, :, 3], data[2, :, 3], data[3, :, 3]]
    lgd = ['ks=3', 'ks=5', 'ks=7', 'ks=9']
    show_fig(d, lgd, 'Epoch', 'Loss', '验证损失', (2, 2, 2))


def show_accuracy():
    d = [data[0, :, 2], data[1, :, 2], data[2, :, 2], data[3, :, 2]]
    lgd = ['ks=3', 'ks=5', 'ks=7', 'ks=9']
    show_fig(d, lgd, 'Epoch', 'Accuracy(%)', '训练精度', (2, 2, 3))

    d = [data[0, :, 4], data[1, :, 4], data[2, :, 4], data[3, :, 4]]
    lgd = ['ks=3', 'ks=5', 'ks=7', 'ks=9']
    show_fig(d, lgd, 'Epoch', 'Accuracy(%)', '验证精度', (2, 2, 4))


if __name__ == '__main__':
    show_loss()
    show_accuracy()
    plt.show()
