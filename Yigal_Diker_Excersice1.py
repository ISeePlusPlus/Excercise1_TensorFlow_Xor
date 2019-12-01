import tensorflow as tf
from pylab import *
import numpy as np


# exercise 1 & 2 involves only n = 2.
# hyper parameters for print
x_train = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_train = [[0.], [1.], [1.], [0.]]
x_validation = [[0., 0.], [0., 1.], [1., 0.], [1., 1.], [1., 0.1], [1., 0.9], [0.9, 0.9], [0.1, 0.9]]
y_validation = [[0.], [1.], [1.], [0.], [1.], [0.], [0.], [1.]]

max_epochs = 40000
improvement_barrier = 0.0001
improvement_barrier_number = 10
loss_barrier = 0.2

# for graphs
hidden_layers_over_mean_epochs = {4: 0., 2: 0.}
bypass_over_mean_epochs = {False: 0, True: 0}
lr_over_std_epochs = {0.1: 0, 0.01: 0}


# using str() to match white spaces in example
def print_hyper_parameters():
    print("Hyper Parameters:\nMax epocs=" + str(max_epochs), "\nstop_Min_loss<" + str(loss_barrier),
          "and loss_Not_Improve_for", improvement_barrier_number,
          "epocs Loss_Min_adv=" + str(improvement_barrier))
    print("\nx_train = " + str(x_train) + "\ny_train = " + str(y_train) + "\nx_valid = " + str(x_validation)
          + "\ny_valid = " + str(y_validation) + "\n\n")


def two_dimensional_xor(x, k, bypass_flag, learning_rate):
    dim = 2
    nb_hidden = k
    nb_outputs = 1
    temperature = 1
    x_train = x
    # y_train = [[0.], [1.], [1.], [0.]]

    x = tf.placeholder(tf.float32, [None, dim])
    target = tf.placeholder(tf.float32, [None, nb_outputs])

    w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1., 1.), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2", dtype=tf.float32)
    z1 = tf.matmul(x, w1) + b1
    h_layer = tf.sigmoid(z1 / temperature)

    if bypass_flag:
        w2 = tf.Variable(tf.random_uniform([dim + nb_hidden, nb_outputs], -1., 1.), dtype=tf.float32)
        h_layer = tf.concat([h_layer, x], 1)
    else:
        w2 = tf.Variable(tf.random_uniform([nb_hidden, nb_outputs], -1., 1.), dtype=tf.float32)

    z2 = tf.matmul(h_layer, w2) + b2
    out = tf.sigmoid(z2/temperature)

    # loss function : crossEntropy
    loss = -tf.reduce_sum(target * tf.log(out) + (1. - target) * tf.log(1. - out))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    min_dict_size = 2
    loss_dict = {}
    counter = 0

    for i in range(max_epochs):
        curr_train, curr_loss = sess.run([train, loss], {x: x_train, target: y_train})
        out_validation, loss_validation = sess.run([out, loss], {x: x_validation, target: y_validation})
        if loss_validation < loss_barrier:
            loss_dict[i] = loss_validation
            if len(loss_dict) >= min_dict_size:
                if loss_dict[i-1] - loss_dict[i] < improvement_barrier:
                    counter += 1
                    if counter is improvement_barrier_number:
                        if nb_hidden == 1:
                            output = sess.run([h_layer], {x: x_train, target: y_train})
                            return output
                        epoch_counter = i
                        return epoch_counter, curr_loss, loss_validation
                else:
                    counter = 0
    return i, curr_loss, loss_validation


def configure_exp(index):
    option_learning_rate = [0.1, 0.01]
    option_num_hidden = [4, 2]
    option_bypass = [False, True]

    num_hidden = option_num_hidden[index % 2]
    bypass = option_bypass[0 if index < 4 else 1]
    lr = option_learning_rate[int(index % 4 / 2)]

    return num_hidden, bypass, lr


def add_points_to_graph(nm_hidden, bypass, learning_rate, mean_epochs, std_epochs):

    hidden_layers_over_mean_epochs[nm_hidden] += mean_epochs
    bypass_over_mean_epochs[bypass] += mean_epochs
    lr_over_std_epochs[learning_rate] += std_epochs


def print_exp(i, nm_hidden, bypass, learning_rate, epochs, loss_train, loss_valid, fails):
    print("experiment " + str(i+1) + ": hidden: " + str(nm_hidden) + ", LR: " + str(learning_rate)
          + ", Bridge:", bypass)

    mean_epochs = np.mean(epochs)
    std_epochs = np.std(epochs)
    # calculating % std
    percentage_std_epochs = (std_epochs / mean_epochs ) * 100
    print(" meanepocs: " + str(mean_epochs) + ", std/epocs% " + str(percentage_std_epochs) + ", Failures:", fails)
    mean_valid = np.mean(loss_valid)
    std_valid = np.std(loss_valid)
    percentage_stdd_valid = (std_valid / mean_valid) * 100
    print(" meanvalidloss: " + str(mean_valid) + ", stdvalidlossPercent: " + str(percentage_stdd_valid) + ",")
    mean_train = np.mean(loss_train)
    std_train = np.std(loss_train)
    percentage_stdd_train = (std_train / mean_train) * 100
    print(" meanTrainLoss: " + str(mean_train) + ", stdTrainLossPercent: " + str(percentage_stdd_train) + "\n ")

    # add points to graphs
    add_points_to_graph(nm_hidden, bypass, learning_rate, mean_epochs, std_epochs)


def make_graph(dict, x_label, y_label, ttl, num_values):
    for value in dict.keys():
        dict[value] /= num_values
    x = list(dict.keys())
    y = list(dict.values())
    plot(x, y)
    xlabel(x_label)
    ylabel(y_label)
    title(ttl)
    show()


def excersice_2():
    print_hyper_parameters()
    num_exp = 8
    num_repeats = 10

    for i in range(num_exp):
        nm_hidden, bypass, learning_rate = configure_exp(i)
        epochs = []
        loss_train = []
        loss_valid = []
        fails = 0
        success = 0
        while success < num_repeats:
            epoch_number, loss_training, loss_validation = two_dimensional_xor\
                (x_train, nm_hidden, bypass, learning_rate)
            if epoch_number == max_epochs - 1.:
                fails += 1
            else:
                epochs.append(epoch_number)
                loss_train.append(loss_training)
                loss_valid.append(loss_validation)
                success += 1
        print_exp(i, nm_hidden, bypass, learning_rate, epochs, loss_train, loss_valid, fails)


    # last experiment
    for index in range(num_repeats):
        output = two_dimensional_xor(x_train, 1, True, 0.1)

        # Changing columns in output
        your_permutation = [2, 0, 1]
        perm_mat = np.zeros((len(your_permutation), len(your_permutation)))
        for idx, i in enumerate(your_permutation):
            perm_mat[idx, i] = 1

        print("experiment 9 run: " + str(index+1))
        print(np.dot(output, perm_mat))

    number_values = 4  # number of values inserted per key
    make_graph(hidden_layers_over_mean_epochs, "Number of hidden nueorons",
               "Mean of epochs", "Hidden layers over Mean epochs", number_values)
    make_graph(bypass_over_mean_epochs, "bypass yes or no",
               "Mean of epochs", "Having bypass over Mean epochs", number_values)
    make_graph(lr_over_std_epochs, "Learning rate",
               "std of epochs", "Learning rate over epochs std", number_values)


excersice_2()
