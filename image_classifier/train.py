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

# Train
import sys
import datetime
from keras.callbacks import TensorBoard
import data_utils
import model

# Config
epochs = 200
workers = 8
data_path = 'tiny-imagenet-200'

# Sample data for generators
# x_samples = data_utils.load_sample_data(data_path)
# for n in x_samples[0][0]:
#     print(n, end=' ')

# print()  # New line

# Build data generators
train_gen, val_gen = data_utils.train_generators(data_path, x_samples=None)
for x_batch, y_batch in train_gen:
    for n in x_batch[0][0]:
        print(n, end=' ')
    print()  # New line
    break

# Build and compile model
train_model, tmpl_model = model.build(gpus=0)

# Tensorboard
run_comment = '0'
if len(sys.argv) > 1:
    run_comment = sys.argv[1]

datetime_str = '{0:%y%m%d_%H%M}'.format(datetime.datetime.now())
tensorboard = TensorBoard(log_dir='logs/' + datetime_str + '_' + run_comment)

# Train
train_model.fit_generator(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    workers=workers,
    use_multiprocessing=True,
    verbose=2,
    callbacks=[tensorboard])

# Save model
tmpl_model.save_weights('weights.h5')
