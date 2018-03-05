# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy
import tensorflow as tf
import model


def main(args):
    # Load training and eval data
    if len(args) < 2:
        print('ERROR: No data path provided')
        return

    base_path = args[1]

    print('Reading data...', end='', flush=True)

    # Init data arrays
    train_data = np.load(os.path.join(base_path, 'train_data.npy'))
    train_labels = np.load(os.path.join(base_path, 'train_labels.npy'))
    eval_data = np.load(os.path.join(base_path, 'eval_data.npy'))
    eval_labels = np.load(os.path.join(base_path, 'eval_labels.npy'))

    print('done', flush=True)
    print('train_data: ', train_data.shape)
    print('eval_data: ', eval_data.shape)

    # Preprocessing
    print('Preprocessing...', end='', flush=True)
    train_data -= np.mean(train_data, axis=0)
    train_data /= np.std(train_data, axis=0)
    eval_data -= np.mean(eval_data, axis=0)
    eval_data /= np.std(eval_data, axis=0)
    print('done', flush=True)

    # Create the Estimator
    imagenet_classifier = tf.estimator.Estimator(model_fn=model.cnn_model_fn,
                                                 model_dir="tiny_imagenet_model")

    # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=50,
        shuffle=True)
    imagenet_classifier.train(input_fn=train_input_fn)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = imagenet_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
