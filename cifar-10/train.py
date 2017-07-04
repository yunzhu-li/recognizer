# Training script

import tflearn
from tflearn.data_utils import shuffle, to_categorical
import network

# Load and preprocess data
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()

X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Build network
nn = network.build_network()

# Create model
model = tflearn.DNN(nn, tensorboard_verbose=0)

# Train model
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')

# Save model to file
model.save("model.tflearn")
