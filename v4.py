import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def prepare_data(k_fold_set, good_thresh, bad_thresh):
    # data prepare
    # traning on pH(legend[8]) and alcohol(legend[10]) parameters
    x_train = []
    y_train = []
    index = 0
    for i in range(len(k_fold_set[1])):
        if int(k_fold_set[1]["quality"][i]) > good_thresh:
            x_train.append([])
            x_train[index].append(1)
            x_train[index].append(float(k_fold_set[1]["pH"][i]))
            x_train[index].append(float(k_fold_set[1]["alcohol"][i]))
            y_train.append(1)
            index += 1
        elif int(k_fold_set[1]["quality"][i]) < bad_thresh:
            x_train.append([])
            x_train[index].append(1)
            x_train[index].append(float(k_fold_set[1]["pH"][i]))
            x_train[index].append(float(k_fold_set[1]["alcohol"][i]))
            y_train.append(-1)
            index += 1
    x_eval = []
    y_eval = []
    index = 0
    for i in range(len(k_fold_set[0])):
        if int(k_fold_set[0]["quality"][i]) > good_thresh:
            x_eval.append([])
            x_eval[index].append(1)
            x_eval[index].append(float(k_fold_set[0]["pH"][i]))
            x_eval[index].append(float(k_fold_set[0]["alcohol"][i]))
            y_eval.append(1)
            index += 1
        elif int(k_fold_set[0]["quality"][i]) < bad_thresh:
            x_eval.append([])
            x_eval[index].append(1)
            x_eval[index].append(float(k_fold_set[0]["pH"][i]))
            x_eval[index].append(float(k_fold_set[0]["alcohol"][i]))
            y_eval.append(-1)
            index += 1
    ret = (x_train, y_train, x_eval, y_eval)
    return ret

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

def k_fold_adaline(k_fold_data, epoch_limit=1000, good_thresh=6, bad_thresh=5, learning_rate=0.0001):

    k_fold_count = -1
    k_len = len(k_fold_data)
    epoch = 0
    cleaned_data = []
    weights = [random.random(), random.random(), random.random()]
    for i in range (k_len):
        cleaned_data.append(prepare_data(k_fold_data[k_fold_count], good_thresh, bad_thresh))
    output_data = []
    # output_data shoul look like [(current_epoch, num_errors_at_epoch_end, [array_of_weights])]

    # training
    while epoch < epoch_limit:
        k_fold_count += 1
        if k_fold_count >= k_len: k_fold_count = 0
        x_train, y_train, x_eval, y_eval = cleaned_data[k_fold_count]

        output = []
        for i in range(len(x_train)):
            output.append(0)
            for j in range(len(x_train[0])):
                output[i] += x_train[i][j] * weights[j]
        errors = []
        for i in range(len(x_train)):
            errors.append(y_train[i] - output[i])
        feedback = [0] * 3
        feedback[0] = sum(errors)

        # updating weights
        t_x_array = transform(x_train)
        for i in range(len(t_x_array) - 1):
            for j in range(len(errors)):
                feedback[i+1] += t_x_array[i+1][j]*errors[j]
        for i in range(len(weights)):
            weights[i] += learning_rate * feedback[i]

        # calculating errors
        result_errors = 0
        predictions = predict(x_train, weights[:])
        for i in range(len(y_train)):
            if y_train[i] != predictions[i]:
                result_errors += 1
        persent_errors = (result_errors * 100)/len(x_train)

        # evaluation errors

        eval_result_errors = 0
        predictions = predict(x_eval, weights[:])
        for i in range(len(y_eval)):
            if y_eval[i] != predictions[i]:
                eval_result_errors += 1
        eval_persent_errors = (eval_result_errors * 100) / len(x_eval)


        output_data.append((epoch, persent_errors, weights[:], eval_persent_errors))

        if result_errors == 0 and eval_result_errors == 0:
            break
        epoch += 1
    return output_data

def plot_preformace_adaline(performance, wine_data, good_thresh, bad_thresh, epoch=-1, save_plot=False):
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
    plt.figure(num=None, figsize=(11, 6), dpi=160, facecolor='w', edgecolor='k')

    # --boundary plot--
    ax = plt.subplot(1, 2, 2)
    plt.xlabel(legend[10])
    plt.ylabel(legend[8])
    plt.title("Decision boundary on epoch: " + str(limit + 1))
    ax.margins(x=0, y=0)

    # draving boundary line
    line_x = [0, x_max]
    line_y = [-weights[0] / weights[1], (-weights[0] - weights[2] * x_max) / weights[1]]
    line = Line2D(line_x, line_y, linewidth=1, color="blue", linestyle="dashed", label='Desidion boundry')
    ax.add_line(line)

    # calculating color for shading areas
    step_function = lambda x: 0 if x < 0 else 1
    result = weights[0] + y_min*weights[1] + x_max*weights[2]
    if step_function(result) == 0:
        down_col = '#ffd7ff'
        up_col = '#d5edd8'
    else:
        down_col = '#d5edd8'
        up_col = '#ffd7ff'

    # shading areas
    ax.fill_between(line_x, y_min, line_y, facecolor=down_col)
    # ax.fill_betweenx(line_x, x_min, line_y, facecolor=down_col)
    ax.fill_between(line_x, y_max, line_y, facecolor=up_col)
    # ax.fill_betweenx(x_max_intercept, x_max, x_intercept, facecolor=up_col)

    # draving dots
    if len(bad_data) > 0:
        ax.plot(bad_data[1], bad_data[0], 'o', c='r', ms=3, label = 'bad wines (<' + str(bad_thresh) + ' score)')
    if len(good_data) > 0:
        ax.plot(good_data[1], good_data[0], 'o', c='g', ms=3, label='good wines (>' + str(good_thresh) + ' score)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # draving legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc = 'upper right', bbox_to_anchor=(1.7, 1))

    # --errors of eposh plot--
    ax1 = plt.subplot(1, 2, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Error %")
    plt.title("Errors as a function of epoch")
    errors = []
    epochs = []
    eval_errors = []

    # drawing plot
    for elem in performance[:limit+1]:
        errors.append(elem[1])
        epochs.append(elem[0])
        eval_errors.append((elem[3]))
    ax1.plot(epochs, errors, label = 'train errors')
    ax1.plot(epochs, eval_errors, c='r', label = 'eval errors')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower left')

    #save show plot
    if save_plot:
        plt.savefig("v3.png")
    plt.show()

if __name__ == '__main__':
    # data = holdout_prepare("./resources/winequality-red.csv", ratio=0.8)
    data2 = k_fold_prepare("./resources/winequality-red.csv", k=8, shuffle=True)
    performance = k_fold_adaline(data2, epoch_limit=20000, good_thresh=6, bad_thresh=5, learning_rate=0.00001)
    plot_preformace_adaline(performance, "./resources/winequality-red.csv", good_thresh=6, bad_thresh=5, save_plot=False)
