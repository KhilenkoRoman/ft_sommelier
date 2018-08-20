import csv
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

random.seed(12345)

def perceptron(wine_data, epoch_limit=1000, good_thresh=7, bad_thresh=4, learning_rate=0.2):
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
    # traning on pH(legend[8]) and alcohol(legend[10]) and residual sugar(legend[3]) parameters
    legend = raw_data.pop(0)
    data = []
    index = 0
    for i in range(len(raw_data)):
        if int(raw_data[i][11]) > good_thresh:
            data.append([])
            data[index].append(1)
            data[index].append(float(raw_data[i][8]))
            data[index].append(float(raw_data[i][10]))
            data[index].append(float(raw_data[i][3]))
            data[index].append(1)
            index += 1
        elif int(raw_data[i][11]) < bad_thresh:
            data.append([])
            data[index].append(1)
            data[index].append(float(raw_data[i][8]))
            data[index].append(float(raw_data[i][10]))
            data[index].append(float(raw_data[i][3]))
            data[index].append(0)
            index += 1

    # prepare vars and step_function for training
    weights = [random.random(), random.random(), random.random(), random.random()]
    epoch = 0
    rlenth = range(len(weights))
    step_function = lambda x: 0 if x < 0 else 1
    output_data = []
    # output_data shoul look like [(current_epoch, num_errors_at_epoch_end, [array_of_weights]), ...]

    # training
    while epoch < epoch_limit:
        errors = 0
        for tmp_data in data:
            result = 0
            for j in rlenth:
                result += float(tmp_data[j]) * weights[j]

            # calculating error
            error = int(tmp_data[4]) - step_function(result)
            if error != 0:
                errors += 1

            # updating weights
            for t in rlenth:
                weights[t] += float(tmp_data[t]) * error * learning_rate

        output_data.append((epoch, errors, weights[:]))
        if errors == 0:
            break
        epoch += 1
    return output_data

def transform(input_matrix):
    res = list(zip(*input_matrix))
    return res

def predict(input_list, weights_list):
    output = []
    prediction = []
    step_function = lambda x: -1 if x < 0 else 1
    for i in range(len(input_list)):
        output.append(0)
        for j in range(len(input_list[0])):
            for j in range(len(input_list[0])):
                output[i] += input_list[i][j] * weights_list[j]
    for pr in output:
        prediction.append(step_function(pr))
    return prediction


def adaline(wine_data, epoch_limit=1000, good_thresh=7, bad_thresh=4, learning_rate=0.0001):
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
            x_array[index].append(1)
            x_array[index].append(float(raw_data[i][8]))
            x_array[index].append(float(raw_data[i][10]))
            x_array[index].append(float(raw_data[i][3]))
            y_array.append(1)
            index += 1
        elif int(raw_data[i][11]) < bad_thresh:
            x_array.append([])
            x_array[index].append(1)
            x_array[index].append(float(raw_data[i][8]))
            x_array[index].append(float(raw_data[i][10]))
            x_array[index].append(float(raw_data[i][3]))
            y_array.append(-1)
            index += 1

    # prepare vars and step_function for training
    weights = [random.random(), random.random(), random.random(), random.random()]
    epoch = 0
    output_data = []
    # output_data shoul look like [(current_epoch, num_errors_at_epoch_end, [array_of_weights]), summ_squared_error]

    # training
    while epoch < epoch_limit:
        output = []
        for i in range(len(x_array)):
            output.append(0)
            for j in range(len(x_array[0])):
                output[i] += x_array[i][j] * weights[j]
        errors = []
        for i in range(len(x_array)):
            errors.append(y_array[i] - output[i])
        feedback = [0] * len(weights)
        feedback[0] = sum(errors)

        # updating weights
        t_x_array = transform(x_array)
        for i in range(len(t_x_array) - 1):
            for j in range(len(errors)):
                feedback[i+1] += t_x_array[i+1][j]*errors[j]
        for i in range(len(weights)):
            weights[i] += learning_rate * feedback[i]
        cost = 0
        for error in errors:
            cost += error**2
        cost = cost / 2.0

        # calculating errors
        result_errors = 0
        predictions = predict(x_array, weights[:])
        for i in range(len(y_array)):
            if y_array[i] != predictions[i]:
                result_errors += 1

        output_data.append((epoch, result_errors, weights[:], cost))
        if result_errors == 0:
            break
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

    #prepearing data for boundary plot
    legend = raw_data.pop(0)
    good_data = []
    bad_data = []
    index_g = 0
    index_b = 0
    for i in range(len(raw_data)):
        if int(raw_data[i][11]) > good_thresh:
            good_data.append([])
            good_data[index_g].append(float(raw_data[i][8]))
            good_data[index_g].append(float(raw_data[i][10]))
            index_g += 1
        elif int(raw_data[i][11]) < bad_thresh:
            bad_data.append([])
            bad_data[index_b].append(float(raw_data[i][8]))
            bad_data[index_b].append(float(raw_data[i][10]))
            index_b += 1

    # matrix transform (.T)
    good_data = list(zip(*good_data))
    bad_data = list(zip(*bad_data))

    # calculating plot sizes
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    if len(good_data) > 0 and len(bad_data) > 0:
        if max(good_data[1]) > max(bad_data[1]):
            x_max = max(good_data[1])
        else:
            x_max = max(bad_data[1])
        if min(good_data[1]) < min(bad_data[1]):
            x_min = min(good_data[1])
        else:
            x_min = min(bad_data[1])
    if len(good_data) > 0 and len(bad_data) > 0:
        if max(good_data[0]) > max(bad_data[0]):
            y_max = max(good_data[0])
        else:
            y_max = max(bad_data[0])
        if min(good_data[0]) < min(bad_data[0]):
            y_min = min(good_data[0])
        else:
            y_min = min(bad_data[0])
    x_min = x_min - x_min * 0.05
    x_max = x_max + x_max * 0.05
    y_min = y_min - y_min * 0.05
    y_max = y_max + y_max * 0.05

    # calculating epoch limit
    len_p = len(performance)
    if epoch <= 0:
        limit = len_p-1
    elif epoch >= len_p:
        limit = len_p-1
    else:
        limit = epoch
    weights = performance[limit][2]

    # main plot
    plt.figure(num=None, figsize=(4, 4), dpi=160, facecolor='w', edgecolor='k')

    # --errors of eposh plot--
    ax1 = plt.subplot(1, 1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Clasification error")
    plt.title("Errors as a function of epoch")
    errors = []
    epochs = []

    # drawing plot
    for elem in performance[:limit+1]:
        errors.append(elem[1])
        epochs.append(elem[0])
    ax1.plot(epochs, errors)

    #save show plot
    if save_plot:
        plt.savefig("v2.png")
    plt.show()


if __name__ == '__main__':
    # performance = perceptron("./resources/winequality-red.csv", epoch_limit=20000, learning_rate=0.0001)
    performance = adaline("./resources/winequality-red.csv", epoch_limit=20000, learning_rate=0.0001)
    plot_preformace(performance, "./resources/winequality-red.csv", good_thresh=7, bad_thresh=4, save_plot=False)