# Classification script

import os
import argparse
import numpy
import scipy
import tflearn
import network

# Reduce Tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parse argument
parser = argparse.ArgumentParser(description='CIFAR-10 image classifier')
parser.add_argument('image', type=str, help='The image file')
args = parser.parse_args()

# Build network
nn = network.build_network()

# Create model and load model
model = tflearn.DNN(nn, tensorboard_verbose=0)
model.load("model/model.tflearn")

# Load the image file
img = scipy.ndimage.imread(args.image, mode="RGB")

# Scale the image to 32x32 and convert to numbers
img = scipy.misc.imresize(img, (32, 32), interp="bicubic")
img = img.astype(numpy.float32, casting='unsafe')

# Predict
prediction = model.predict([img])

# Output
names = ["airplane", "automobile", "bird", "cat",
         "deer", "dog", "frog", "horse", "ship", "truck"]

# Best probability
print("\nMost likely: " + names[numpy.argmax(prediction[0])])
