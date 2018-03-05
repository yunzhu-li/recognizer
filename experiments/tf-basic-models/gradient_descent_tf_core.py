#!/usr/bin/env python3
import tensorflow as tf

# Define model
W = tf.Variable([30], dtype=tf.float16)
b = tf.Variable([2], dtype=tf.float16)
x = tf.placeholder(tf.float16)
y = tf.placeholder(tf.float16)
linear_model = W * x + b

# Loss
squared_deltas = tf.square(linear_model - y)
loss_func = tf.reduce_sum(squared_deltas)

# Trainer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss_func)

# Define training data
x_train = [1, 2, 3, 4]
y_train = [7, 8, 9, 10]

# Init vars
init_vars = tf.global_variables_initializer()

# Init session
sess = tf.Session()
sess.run(init_vars)

# Print loss
print('Initial loss:')
print(sess.run(loss_func, {x: x_train, y: y_train}))

last_loss = 0
while True:
    # Run GD
    sess.run(train, {x: x_train, y: y_train})

    # Calculate loss
    loss = sess.run(loss_func, {x: x_train, y: y_train})

    # Stop condition
    if (abs(loss - last_loss) < 0.001):
        break;

    last_loss = loss

# Use training data to validate in this example
print('Final loss:')
print(sess.run(loss_func, {x: x_train, y: y_train}))
