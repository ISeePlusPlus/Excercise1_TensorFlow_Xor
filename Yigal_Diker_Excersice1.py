import tensorflow as tf


def print_details(k, curr_w1, curr_b1, curr_w2, curr_b2, curr_out, curr_loss):
    print("Results for k = ", k)
    print("w1 = \n", curr_w1)
    print("b1 = ", curr_b1)
    print("w2 = \n", curr_w2)
    print("b2 = ", curr_b2)
    print("Result: \n", curr_out)
    print("Loss: ", curr_loss)


def two_dimensional_xor(x, k, bypass_flag, learning_rate):

    dim = 2
    nb_hidden = k
    nb_outputs = 1
    temperature = 1
    x_train = x
    y_train = [[0.], [1.], [1.], [0.]]

    x = tf.placeholder(tf.float32, [None, dim])
    target = tf.placeholder(tf.float32, [None, nb_outputs])

    w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1., 1., seed=0))
    b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")
    b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")
    z1 = tf.matmul(x, w1) + b1
    h_layer = tf.sigmoid(z1 / temperature)

    if bypass_flag:
        w2 = tf.Variable(tf.random_uniform([dim + nb_hidden, nb_outputs], -1., 1., seed=0))
        h_layer = tf.concat([h_layer, x], 1)
    else:
        w2 = tf.Variable(tf.random_uniform([nb_hidden, nb_outputs], -1., 1., seed=0))

    z2 = tf.matmul(h_layer, w2) + b2
    out = tf.sigmoid(z2/temperature)

    # loss function : crossEntropy
    loss = -tf.reduce_sum(target * tf.log(out) + (1. - target) * tf.log(1. - out))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # validation group
    x_validation = [[0., 0.], [0., 1.], [1., 0.], [1., 1.], [1., 0.1], [1., 0.9], [0.9, 0.9], [0.1, 0.9]]
    y_validation = [[0.], [1.], [1.], [0.], [1.], [0.], [0.], [1.]]

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    max_epochs = 40000
    min_dict_size = 2
    improvement_barrier = 0001.0
    improvement_barrier_number = 10
    loss_barrier = 0.2
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
                        print(i)
                        break
                else:
                    counter = 0


# exercise 1 involves only n = 2
x_dim2 = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
bypass = True
option_learning_rate = 0.1

two_dimensional_xor(x_dim2, 4, False, option_learning_rate)
two_dimensional_xor(x_dim2, 2, False, option_learning_rate)
two_dimensional_xor(x_dim2, 1, bypass, option_learning_rate)
