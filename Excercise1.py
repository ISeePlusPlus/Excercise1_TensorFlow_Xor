import tensorflow as tf

sess = tf.Session()

# Defining a sigmoid.
w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

x = tf.compat.v1.placeholder(tf.float32, [None], name="x_linear")
z = tf.multiply(x, w) + b
Temp = 0.001

# sigmoid = 1 / (1 + e^(-z/Temp))
out = tf.sigmoid(z / Temp, "sigmoid")  # name is sigmoid
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print(sess.run(out, {x: [1, 2, 3, 4]}))

# Defining a loss function. # optional depends if Gadi taught ML

# This loss function is the squre sum of the difference between t and the linear model output.
t = tf.compat.v1.placeholder(tf.float32)  # t is the target vector.
squared_deltas = tf.square(out - t)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], t: [0, -1, -2, -3]}))

# simple 4 neurons hidden layer.
dim = 2  # Input dim
num_outputs = 1
num_hidden = 4
Temp = 0.001  # for sigmoid
x = tf.compat.v1.placeholder(tf.float32, [None, dim], name="net_input")
w1 = tf.Variable(tf.random.uniform([dim, num_hidden], -1, 1, seed=0), name="weights1")
b1 = tf.Variable(tf.zeros(num_hidden), name="Biases1")
z1 = tf.matmul(x, w1) + b1  # [4, 2] X [2, num_hidden] results in a vector [4, num_hidden] of z's Theoretically
# # [1, num_hidden] for every input pair, which is the output of the hidden layer (before sigmoid).
hlayer = tf.sigmoid(z1/Temp)  # Element wise

init = tf.compat.v1.global_variables_initializer()
sess = tf.Session()
sess.run(init)
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
print(sess.run(hlayer, {x: x_train}))

# In order to create bypass synapse use tf.concat, use tensorflow documantation for further information.
# In fact we have to create wieghts matrix combining the wieghts of the hidden layers synapses and the inputs (placeholder) synapses.

dim=2
nb_outputs=1
nb_hidden=4
temp=0.001

x = tf.placeholder(tf.float32, [None, dim])

w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1, 1, seed=0),     name="Weights1")
w2 = tf.Variable(tf.random_uniform([nb_hidden, nb_outputs], -1, 1,   seed=0), name="Weights2")
b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")
b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")
z1=tf.matmul(x, w1) + b1	# [4,2]X[2,nb_hidden]  results in a vector [4,nb_hidden] of z’s
hlayer1 = tf.sigmoid(z1/temp)  # element wise
z2=tf.matmul(hlayer1, w2) + b2
out = tf.sigmoid(z2/temp)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
x_train = [[0,0],[0,1],[1,0],[1,1]]
print(sess.run(out, {x:x_train}))


dim=2
nb_hidden=1
nb_outputs=1
t=0.001      #cold temperature (sharp slop of the sigmoid
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[1],[1],[0]]
nb_hbridge = nb_hidden+ dim
x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32,[None,1])
w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1, 1, seed=0))
w2 = tf.Variable(tf.random_uniform([nb_hbridge, nb_outputs], -1, 1,  seed=0))
b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")
b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")
z1=tf.matmul(x, w1) + b1
hlayer = tf.sigmoid(z1/t)
hlayer1=tf.concat([hlayer, x],1)
z2=tf.matmul(hlayer1, w2) + b2
out = tf.sigmoid(z2/t)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # initialize variables
curr_w1, curr_b1, curr_w2, curr_b2, curr_hlayer, curr_out, curr_loss = sess.run([w1, b1, w2, b2,    hlayer1,  out, loss],   {x:x_train,  y:y_train})
