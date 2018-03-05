# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.ndimage
import sys

# Configurations
num_classes = 20
images_per_class = 450
train_images_per_class = 400

# Load training and eval data
if len(sys.argv) < 2:
    print('ERROR: No data path provided')
    sys.exit(1)

# Init data lists
# Can also initialize as arrays like:
# train_data = np.empty((0, 64, 64, 3), dtype=np.float32)
# but append operation is slow since it copies data.
train_data = []
train_labels = []
eval_data = []
eval_labels = []

# Get a list of subdirectories in base_path, each is one class
base_path = sys.argv[1]
train_data_path = os.path.join(base_path, 'train')
class_paths = []
for _, dirs, _ in os.walk(train_data_path):
    class_paths = dirs
    break  # Only need first level

# Sort paths
class_paths.sort()

# Read images and assign labels
class_id = 0
for class_path in class_paths:
    # Print status
    print('Reading class: {0} ({1}/{2})'.format(class_path, class_id + 1, num_classes))

    # Images base path
    img_path_full = os.path.join(train_data_path, class_path, 'images')

    # All images
    files = os.listdir(img_path_full)
    files = [os.path.join(img_path_full, f) for f in files]

    for i in range(0, images_per_class):
        # Decode image
        img = scipy.ndimage.imread(files[i], mode='RGB')
        img = img.reshape(64, 64, 3).tolist()

        if i < train_images_per_class:
            # Read first some files as training data
            train_data.append(img)
            train_labels.append(class_id)
        else:
            # Read some more for evaluation data
            eval_data.append(img)
            eval_labels.append(class_id)

    class_id += 1
    if class_id >= num_classes:
        break

# Convert back to numpy arrays
print('Converting to numpy array...', end='', flush=True)
train_data = np.asarray(train_data, dtype=np.float32)
train_labels = np.asarray(train_labels, dtype=np.int32)
eval_data = np.asarray(eval_data, dtype=np.float32)
eval_labels = np.asarray(eval_labels, dtype=np.int32)
print('done', flush=True)

print('train_data:', train_data.shape, 'train_labels:', train_labels.shape)
print('eval_data:', eval_data.shape, 'eval_data:', eval_labels.shape)

print('Saving arrays...', end='', flush=True)
np.save(os.path.join(base_path, 'train_data.npy'), train_data)
np.save(os.path.join(base_path, 'train_labels.npy'), train_labels)
np.save(os.path.join(base_path, 'eval_data.npy'), eval_data)
np.save(os.path.join(base_path, 'eval_labels.npy'), eval_labels)
print('done', flush=True)
