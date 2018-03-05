'''
yl-recognizer
Copyright (C) 2017-2018 Yunzhu Li

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

# CNN model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import metrics
from keras.utils import multi_gpu_model


# Load saved model
def load(weights_path):
    model, tmpl_model = build(gpus=0)
    model.load_weights(weights_path)
    return model, tmpl_model


# Build model
def build(gpus=1):
    # Create new template model
    tmpl_model = Sequential()

    # In: [-1, 64, 64, 3]
    tmpl_model.add(Conv2D(filters=96, kernel_size=[5, 5], padding='same', input_shape=(64, 64, 3)))
    tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(MaxPooling2D(pool_size=[2, 2]))

    # In: [-1, 32, 32, 96]
    tmpl_model.add(Conv2D(128, [3, 3], padding='same'))
    tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(Conv2D(192, [3, 3], padding='same'))
    tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(MaxPooling2D([2, 2]))

    # In: [-1, 16, 16, 192]
    tmpl_model.add(Conv2D(256, [3, 3], padding='same'))
    tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(Conv2D(256, [3, 3], padding='same'))
    tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(Conv2D(128, [3, 3], padding='same', activation='relu'))
    tmpl_model.add(MaxPooling2D([2, 2]))

    # Flatten to 1-D vector
    # In: [-1, 8, 8, 192]
    tmpl_model.add(Flatten())

    # Dense
    tmpl_model.add(Dense(units=1024, activation='relu'))

    # Dropout
    tmpl_model.add(Dropout(0.5))

    # Dense
    tmpl_model.add(Dense(units=512, activation='relu'))

    # Logits layer
    tmpl_model.add(Dense(units=200, activation='softmax'))

    # Optimizers
    adadelta = optimizers.Adadelta()
    adam = optimizers.Adam(lr=0.001)
    rmsprop = optimizers.RMSprop(lr=0.001)

    # lr = self.lr * (1. / (1. + self.decay * self.iterations))
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    # The model to be trained
    train_model = tmpl_model

    if gpus > 1:
        # Train on parallel model
        train_model = multi_gpu_model(tmpl_model, gpus=gpus)

    # Compile model
    train_model.compile(optimizer=sgd,
                        loss='categorical_crossentropy',
                        metrics=['accuracy', metrics.top_k_categorical_accuracy])

    return train_model, tmpl_model
