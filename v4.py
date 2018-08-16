import csv
import random
import pandas as pd

def transform(input_matrix):
	res = list(zip(*input_matrix))
	return res

def holdout_prepare(wine_data, ratio=0.8):
	try:
		with open(wine_data, newline='') as csvfile:
			raw_data = list(csv.reader(csvfile, delimiter=';'))
	except FileNotFoundError as err:
		print(err.args)
		return 0
	except Exception:
		print('dich')
		return 0
	if ratio <= 0 or ratio >= 1:
		raise ValueError('Wrong ratio')


	legend = raw_data.pop(0)
	data_len = len(raw_data)
	data_slice = int(data_len*ratio)

	raw_data = transform(raw_data)
	d = {}
	for i in range(len(legend)):
		d[legend[i]] = raw_data[i][:data_slice]
	df1 = pd.DataFrame(data=d)
	d = {}
	for i in range(len(legend)):
		d[legend[i]] = raw_data[i][data_slice:]
	df2 = pd.DataFrame(data=d)
	ret = (df1, df2)
	return ret

def k_fold_prepare(wine_data, k=4, shuffle=False):
	try:
		with open(wine_data, newline='') as csvfile:
			raw_data = list(csv.reader(csvfile, delimiter=';'))
	except FileNotFoundError as err:
		print(err.args)
		return 0
	except Exception:
		print('dich')
		return 0
	if k <= 0:
		raise ValueError('k')


	legend = raw_data.pop(0)
	if shuffle:
		random.shuffle(raw_data)
	data_len = len(raw_data)
	data_partition = []
	p = int(data_len / k)
	for i in range(k):
		data_partition.append(raw_data[i*p:(i+1)*p])

	output = []
	for i in range(k):
		traning = []
		test = []
		for j in range(k):
			if j == i:
				traning = data_partition[j]
			else:
				test.extend(data_partition[j])
		traning = transform(traning)
		test = transform(test)

		d = {}
		for i in range(len(legend)):
			d[legend[i]] = traning[i]
		df1 = pd.DataFrame(data=d)
		d = {}
		for i in range(len(legend)):
			d[legend[i]] = test[i]
		df2 = pd.DataFrame(data=d)
		ret = (df1, df2)
		output.append(ret)
	return output





if __name__ == '__main__':
	# data = holdout_prepare("./resources/winequality-red.csv", ratio=0.8)
	data2 = k_fold_prepare("./resources/winequality-red.csv", k=8, shuffle=True)
	print(data2[0])

