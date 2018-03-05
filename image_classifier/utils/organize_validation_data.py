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

# Move validation data into folders like training data
import os
import shutil

base_path = 'tiny-imagenet-200/val'

with open(os.path.join(base_path, 'val_annotations.txt')) as f:
    lines = f.readlines()
    file_count = 0
    for line in lines:
        file_count += 1
        print('Processing file', file_count)

        # Get file name and it's class name
        img_info = line.split('\t')
        file_name = img_info[0]
        file_class = img_info[1]

        img_directory = os.path.join(base_path, file_class)

        # Make directory using class name
        if not os.path.exists(img_directory):
            os.mkdir(img_directory)

        # Copy file
        shutil.copy(os.path.join(base_path, 'images', file_name),
                    os.path.join(img_directory, file_name))
