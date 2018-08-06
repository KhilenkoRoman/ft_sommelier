import csv
import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):
	try:
		with open(wine_data, newline='') as csvfile:
			data = list(csv.reader(csvfile, delimiter=';'))
	except FileNotFoundError as err:
		print(err.args)
		return 0

	# for parameter in data[0]:
	# 	print(parameter)

	print(data[0])
	data.pop(0)
	# print(data)

	good_list = []
	bad_list = []
	for i in data:
		if int(i[-1]) > good_threshold:
			good_list.append(i)
		if int(i[-1]) < bad_threshold:
			bad_list.append(i)

	good_list = list(zip(*good_list))
	bad_list = list(zip(*bad_list))

	print(len(good_list[2]))

	matplotlib.rcParams['axes.unicode_minus'] = False
	fig, ax = plt.subplots()
	ax.plot(good_list[2], good_list[4], 'o', c='g')
	ax.plot(bad_list[2], bad_list[4], 'o', c='r')
	ax.set_title('Using hyphen instead of Unicode minus')
	plt.show()


plot_scatter_matrix("./resources/winequality-red.csv", 6, 5)
