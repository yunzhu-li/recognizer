# Modified from https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/layers/cnn_mnist.py
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.

# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import tensorflow as tf
import mnist_model


def main(argv):
    # Read image files
    if len(argv) < 2:
        print('No image file(s) provided')
        return

    predict_data = np.empty((0, 28, 28), dtype=np.float32)

    for i in range(1, len(argv)):
        img = scipy.ndimage.imread(argv[i], flatten=True)
        # Scale the image to 28x28 and convert to numbers
        img = scipy.misc.imresize(img, (28, 28), interp='bicubic')
        img = img.astype(np.float32)
        img = img / 255  # Scale to [0, 1]
        img = 1 - img  # Reverse
        img = img.reshape(1, 28, 28)
        predict_data = np.append(predict_data, img, axis=0)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=mnist_model.cnn_model_fn,
                                              model_dir='mnist_model')

    # Evaluate the model and print results
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': predict_data},
        num_epochs=1,
        shuffle=False)
    predict_results = mnist_classifier.predict(input_fn=predict_input_fn)

    for r in predict_results:
        print('class: ', r['classes'])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main)
