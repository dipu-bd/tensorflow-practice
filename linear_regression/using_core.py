"""
Source: https://www.tensorflow.org/get_started/get_started
A complete trainable linear regression model using core TensorFlow elements
"""
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32, name='W')
b = tf.Variable([-.3], tf.float32, name='b')

# Model input and output
x = tf.placeholder(tf.float32, name='x')
linear_model = W * x + b
y = tf.placeholder(tf.float32, name='y')

# loss - sum of the square
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# initialize paramenters
init = tf.global_variables_initializer()
sess = tf.Session()

# define writer
writer = tf.summary.FileWriter('../.bin', sess.graph)

# run the init under session
sess.run(init)

# training loop
print()
print('  x:', x_train)
print('  y:', y_train)
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
cur_W, cur_b, cur_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print('After training:')
print('  W:', cur_W)
print('  b:', cur_b)
print('  loss:', cur_loss)

sess.close()
writer.close()
