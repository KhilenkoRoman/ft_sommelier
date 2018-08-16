import csv
import random
from v2 import plot_preformace
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

random.seed(12345)

def transform(input_matrix):
	res = list(zip(*input_matrix))
	return res


def dot(a,b):
	zip_b = zip(*b)
	zip_b = list(zip_b)
	return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b))
			 for col_b in zip_b] for row_a in a]

class AdalineGD(object):

	def __init__(self, eta=0.01, epochs=50):
		self.eta = eta
		self.epochs = epochs

	def train(self, inp, expext):

		y = np.array(expext)
		X = np.array(inp)

		# self.w_ = np.zeros(1 + X.shape[1])
		self.w_ = [random.random(), random.random(), random.random()]
		self.cost_ = []

		for i in range(self.epochs):
			output = np.dot(X, self.w_[1:]) + self.w_[0]
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()

			# [0.38997930067274705, -0.07845150454852957, 0.5210810914812147]
			cost = (errors**2).sum() / 2.0
			# 35486.00348494256
			self.cost_.append(cost)
		return self

	def print_weights(self):
		print(self.w_)

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, 1, -1)

def adaline(wine_data, epoch_limit=1000, good_thresh=7, bad_thresh=4, learning_rate=0.2):
	try:
		with open(wine_data, newline='') as csvfile:
			raw_data = list(csv.reader(csvfile, delimiter=';'))
	except FileNotFoundError as err:
		print(err.args)
		return 0
	except Exception:
		print('dich')
		return 0

	# data prepare
	# traning on pH(legend[8]) and alcohol(legend[10]) parameters
	legend = raw_data.pop(0)
	x_array = []
	y_array = []
	index = 0
	for i in range(len(raw_data)):
		if int(raw_data[i][11]) > good_thresh:
			x_array.append([])
			x_array[index].append(float(raw_data[i][8]))
			x_array[index].append(float(raw_data[i][10]))
			y_array.append(1)
			index += 1
		elif int(raw_data[i][11]) < bad_thresh:
			x_array.append([])
			x_array[index].append(float(raw_data[i][8]))
			x_array[index].append(float(raw_data[i][10]))
			y_array.append(-1)
			index += 1

	# x_array = np.array(x_array)
	# y_array = np.array(y_array)

	# df = pd.read_csv('iris.data', header=None)
	# # setosa and versicolor
	# y = df.iloc[0:100, 4].values
	# y = np.where(y == 'Iris-setosa', -1, 1)
	# x = df.iloc[0:100, [0, 2]].values

	ada = AdalineGD(epochs=1000, eta=0.00001).train(x_array, y_array)
	print(ada.predict(x_array))
	plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
	plt.xlabel('Iterations')
	plt.ylabel('Sum-squared-error')
	plt.title('Adaline - Learning rate 0.0001')
	plt.show()

	# ada.print_weights()





if __name__ == '__main__':
	performance = adaline("./resources/winequality-red.csv", epoch_limit=100, learning_rate=0.01)
	# plot_preformace(performance, "./resources/winequality-red.csv", good_thresh=7, bad_thresh=4, save_plot=True)