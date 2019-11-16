import tensorflow as tf


def print_details(k, curr_w1, curr_b1, curr_w2, curr_b2, curr_out, curr_loss):
    print("Results for k = ", k)
    print("w1 = \n", curr_w1)
    print("b1 = ", curr_b1)
    print("w2 = \n", curr_w2)
    print("b2 = ", curr_b2)
    print("Result: \n", curr_out)
    print("Loss: ", curr_loss)


def two_dimensional_xor(x_dim2, k):

    dim = 2
    nb_hidden = k
    nb_outputs = 1
    temperature = 0.001      # cold temperature (sharp slop of the sigmoid
    x_train = x_dim2
    y_train = [[0.], [1.], [1.], [0.]]

    x = tf.placeholder(tf.float32, [None, dim])
    y = tf.placeholder(tf.float32, [None, nb_outputs])
    w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1, 1, seed=0))
    w2 = tf.Variable(tf.random_uniform([nb_hidden, nb_outputs], -1, 1, seed=0))
    b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")
    b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")
    z1 = tf.matmul(x, w1) + b1
    hlayer = tf.sigmoid(z1/temperature)
    z2 = tf.matmul(hlayer, w2) + b2
    out = tf.sigmoid(z2/temperature)

    # loss function
    target = tf.compat.v1.placeholder(tf.float32)
    squared_deltas = tf.square(out - target)
    loss = tf.reduce_sum(squared_deltas)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) # initialize variables
    curr_w1, curr_b1, curr_w2, curr_b2, curr_out, curr_loss = \
        sess.run([w1, b1, w2, b2, out, loss], {x:x_train,  target: y_train})
    print("Before assignment: ")
    print_details(k, curr_w1, curr_b1, curr_w2, curr_b2, curr_out, curr_loss)

    # k can be any number, for this exercise we use 4, 2 and 1. also can be any dimension, for this exercise it is 2

    if nb_hidden == 4:
        fix_w1 = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
        # transpose messes up the print a bit as w1 is printed transposed, this is correct w1 vector before transpose
        fix_w1 = tf.transpose(fix_w1)
        fix_w1 = tf.compat.v1.assign(w1, fix_w1)
        fix_w2 = tf.compat.v1.assign(w2, [[-1.], [1.], [1.], [-1.]])
        fix_b1 = tf.compat.v1.assign(b1, [0.5, -.5, -.5, -1.5])
        fix_b2 = tf.compat.v1.assign(b2, [-.5])
    elif nb_hidden == 2:
        fix_w1 = tf.compat.v1.assign(w1, [[-1., 1.], [1., -1.]])
        fix_w2 = tf.compat.v1.assign(w2, [[1.], [1.]])
        fix_b1 = tf.compat.v1.assign(b1, [-.5, -.5])
        fix_b2 = tf.compat.v1.assign(b2, [-.5])
    else:  # nb_hidden = 1
        w2 = tf.Variable(tf.random_uniform([dim + nb_hidden, nb_outputs], -1, 1,  seed=0))
        hlayer1 = tf.concat([hlayer, x], 1)
        init = tf.global_variables_initializer()
        sess.run(init)

        # Overriding previous declarations
        z2 = tf.matmul(hlayer1, w2) + b2
        out = tf.sigmoid(z2/temperature)
        squared_deltas = tf.square(out - target)
        loss = tf.reduce_sum(squared_deltas)

        fix_w1 = tf.compat.v1.assign(w1, [[1.], [1.]])
        fix_w2 = tf.compat.v1.assign(w2, [[-2], [1.], [1.]])
        fix_b1 = tf.compat.v1.assign(b1, [-1.5])
        fix_b2 = tf.compat.v1.assign(b2, [-.5])
        sess.run([fix_w1, fix_b1, fix_w2, fix_b2, out, loss], {x: x_train, target: y_train})

    curr_w1, curr_b1, curr_w2, curr_b2, curr_out, curr_loss = \
        sess.run([fix_w1, fix_b1, fix_w2, fix_b2, out, loss], {x: x_train,  target: y_train})
    print("After assignment: ")
    print_details(k, curr_w1, curr_b1, curr_w2, curr_b2, curr_out, curr_loss)


# exercise 1 involves only n = 2
x_dim2 = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]

two_dimensional_xor(x_dim2, 4)
two_dimensional_xor(x_dim2, 2)
two_dimensional_xor(x_dim2, 1)
