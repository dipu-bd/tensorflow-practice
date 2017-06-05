"""
Source: https://www.tensorflow.org/get_started/get_started
A complete trainable linear regression model using the
tf.contrib.learn library.
"""
import numpy as np
import tensorflow as tf

# Declare list of features
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# Estimator is the front-end to invoke training (fitting), and
# evaluation (inference). An estimator for linear regression is
# declared below
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Build the input function
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)

# training process
estimator.fit(input_fn=input_fn, steps=1000)

# evaluation
print()
print(estimator.evaluate(input_fn=input_fn))
