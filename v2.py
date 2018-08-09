import csv
import random
from random import choice
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

random.seed(12345)


def perceptron(wine_data, epoch_limit=1000, good_thresh=8, bad_thresh=3, learning_rate=0.2):
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
	data = []

	# traning on pH(legend[8]) and alcohol(legend[10]) parameters
	index = 0
	for i in range(len(raw_data)):
		if int(raw_data[i][11]) >= good_thresh:
			data.append([])
			data[index].append(1)
			data[index].append(float(raw_data[i][8]))
			data[index].append(float(raw_data[i][10]))
			data[index].append(1)
			index += 1
		elif int(raw_data[i][11]) <= bad_thresh:
			data.append([])
			data[index].append(1)
			data[index].append(float(raw_data[i][8]))
			data[index].append(float(raw_data[i][10]))
			data[index].append(0)
			index += 1
	step_function = lambda x: 0 if x < 0 else 1

	weights = [random.random(), random.random(), random.random()]
	epoch = 0
	rlenth = range(len(weights))
	output_data = []
	# [(current_epoch, num_errors_at_epoch_end, [array_of_weights]), ...]

	while epoch < epoch_limit:
		errors = 0
		for tmp_data in data:
			result = 0
			for j in rlenth:
				result += float(tmp_data[j]) * weights[j]
			error = int(tmp_data[3]) - step_function(result)
			# print("expect ", int(tmp_data[3]), " get ", step_function(result), " == ", error)
			if error != 0:
				errors += 1
			for t in rlenth:
				weights[t] += float(tmp_data[t]) * error * learning_rate

		output_data.append((epoch, errors, weights))
		epoch += 1
	return output_data


def plot_preformace(performance, wine_data, good_thresh, bad_thresh, epoch=-1, save_plot=False):
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

	good_data = list(zip(*good_data))
	bad_data = list(zip(*bad_data))

	plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

	ax = plt.subplot(1, 2, 2)

	# xcoords = [0.22058956, 0.33088437, 2.20589566]
	# for xc in xcoords:
	x = [1, 2]
	y = [1, 2]
	# ax.plot(x, y)

	ax.fill_between(x, 2, y, facecolor='#ffd7ff')
	ax.fill_between(x, 1, y, facecolor='#d5edd8')

	if len(good_data) > 0:
		ax.plot(good_data[1], good_data[0], 'o', c='g', ms=3)
	if len(bad_data) > 0:
		ax.plot(bad_data[1], bad_data[0], 'o', c='r', ms=3)

	plt.show()


if __name__ == '__main__':
	performance = perceptron("./resources/winequality-white.csv", epoch_limit=1000, learning_rate=0.2)
	plot_preformace(performance, "./resources/winequality-white.csv", good_thresh=8, bad_thresh=3, save_plot=True)
