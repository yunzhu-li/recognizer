''' Convolutional network applied to CIFAR-10 dataset classification task

References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Dataset:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

Forked from:
    https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
'''

from tflearn import *


# Build network
def build_network():
    # Data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Data augmentation
    # Apply transformations to input images to create more data
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network
    # 32x32 image with 3 color channels
    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network
