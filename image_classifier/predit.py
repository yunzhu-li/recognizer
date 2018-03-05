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

# Run predition on images in test_images
import sys
import numpy as np
from keras.models import load_model
import data_utils
import model

# Config
# Trained model
trained_model_path = 'trained_models/180301_tiny_imagenet_weights.h5'

# Test data path
test_images_path = 'test_images'

# Build data generators
predict_gen = data_utils.predict_generator(test_images_path, x_samples=None)

# Read data
predit_x, filenames = data_utils.read_files_into_memory(predict_gen)
print(predit_x[0][0])

# Build and compile model
print('Loading model...')
predict_model, _ = model.load(trained_model_path)

# Predit
print('Predicting...')
preds = predict_model.predict(predit_x, batch_size=1, verbose=1)
results = data_utils.decode_predictions(preds)

# Iterate through results
for i in range(len(results)):
    filename = filenames[i]
    print('\n--- ' + filename)
    print(results[i])
