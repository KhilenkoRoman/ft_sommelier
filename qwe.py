import csv
import random
from random import choice
import matplotlib.pyplot as plt

import numpy as np
from numpy import dot, array

random.seed(12345)


def perceptron(wine_data, epoch_limit=1000000, good_thresh=8, bad_thresh=3, learning_rate=0.2):
    try:
        with open(wine_data, newline='') as csvfile:
            raw_data = list(csv.reader(csvfile, delimiter=';'))
    except FileNotFoundError as err:
        print(err.args)
        return 0
    except Exception:
        print('dich')
        return 0

    legend = raw_data.pop(0)
    good_data = []
    bad_data = []

    # traning on pH(legend[8]) and alcohol(legend[10]) parameters
    index_g = 0
    index_b = 0
    for i in range(len(raw_data)):
        if int(raw_data[i][11]) >= good_thresh:
            good_data.append([])

            good_data[index_g].append(float(raw_data[i][8]))
            good_data[index_g].append(float(raw_data[i][10]))

            index_g += 1
        elif int(raw_data[i][11]) <= bad_thresh:
            bad_data.append([])

            bad_data[index_b].append(float(raw_data[i][8]))
            bad_data[index_b].append(float(raw_data[i][10]))

            index_b += 1

    print(good_data)
    print(bad_data)

    num_good_data = array(good_data)
    num_bad_data = array(bad_data)
    print(num_bad_data.T)

    ax = plt.subplot(1, 1, 1)

    ax.plot(num_good_data.T[1], num_good_data.T[0], 'o', c='g', ms=1)
    ax.plot(num_bad_data.T[1], num_bad_data.T[0], 'o', c='r', ms=1)

    # plt.xticks([])
    # plt.yticks([])

    plt.savefig("vinisko.png")
    plt.show()


if __name__ == '__main__':
    perceptron("./resources/winequality-white.csv", epoch_limit=100000, learning_rate=0.2)
