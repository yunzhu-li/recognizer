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

# Data utilities
import io
import os
from os import path
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array


# Create data generators with augmentation from directories
def train_generators(base_path, x_samples):

    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        # zca_whitening=True,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(
        featurewise_center=False,
        rescale=1./255)

    # Fit generator if x samples provided
    if x_samples is not None:
        train_datagen.fit(x_samples / 255)
        val_datagen.fit(x_samples / 255)

    train_gen = train_datagen.flow_from_directory(
        directory=path.join(base_path, 'train'),
        target_size=(64, 64),
        batch_size=512,
        class_mode='categorical')

    val_gen = val_datagen.flow_from_directory(
        directory=path.join(base_path, 'val'),
        target_size=(64, 64),
        batch_size=512,
        class_mode='categorical')

    return train_gen, val_gen


def predict_generator(base_path, x_samples):
    predict_datagen = ImageDataGenerator(
        featurewise_center=False,
        rescale=1./255)

    # Fit generator if x samples provided
    if x_samples is not None:
        predict_datagen.fit(x_samples / 255)

    predict_gen = predict_datagen.flow_from_directory(
        directory=base_path,
        target_size=(64, 64),
        batch_size=512,
        class_mode=None,
        shuffle=False)

    return predict_gen


# Read sample images
def load_sample_data(base_path):
    print('[PROG] Sampling data for fitting ImageDataGenerator...')
    sample_datagen = ImageDataGenerator()
    sample_gen = sample_datagen.flow_from_directory(
        directory=path.join(base_path, 'train'),
        target_size=(64, 64),
        batch_size=1000,
        class_mode='categorical')

    for x_batch, y_batch in sample_gen:
        print('[PROG] Done sampling data')
        return x_batch

    return None


# Read files from a generator, return array and file names
def read_files_into_memory(gen):
    filenames = gen.filenames
    for i in range(len(gen)):
        x_batch = next(gen)
        # Highly inefficient append operation
        if 'predit_x' not in locals():
            predit_x = x_batch
        else:
            predit_x = np.append(predit_x, x_batch, axis=0)

    return predit_x, filenames


# Decode and pre-process image data into numpy array
def image_array_from_bytes(data):
    image = Image.open(io.BytesIO(data))

    # To RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize
    image = image.resize((64, 64))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)

    # Scale
    arr = arr / 255.

    return arr


# Return top 5 classes by probabilities
def decode_predictions(pred):
    # Get class names
    cls_names = class_names()

    # Walk through samples
    results = []
    for i in range(len(pred)):
        probabilities = pred[i]

        # Sort array in descending order, retrieve indexes (class_ids)
        sorted_idx = np.argsort(probabilities)[::-1]

        # Adding top 5 results
        results.append([])
        k = 0
        for clsid in sorted_idx:
            results[i].append({
                'class_id': int(clsid),
                'class_name': cls_names[clsid],
                'probability': float(probabilities[clsid])
            })

            # Only top 5
            k += 1
            if k >= 5:
                break

    return results


# Build a list of class names corresponding to alphabetical order of
# directories in base_path/train
# Works with Stanford CS231N Tiny ImageNet directory structure
# https://tiny-imagenet.herokuapp.com/
def build_class_names(base_path):
    # Build full id-word mapping from words.txt
    full_dict = {}
    f = open(os.path.join(base_path, 'words.txt'))
    lines = f.readlines()
    for line in lines:
        pair = line.split('\t')
        full_dict[pair[0]] = pair[1].strip()

    # Match class names and add to list
    class_names = []
    for _, dirs, _ in os.walk(os.path.join(base_path, 'train')):
        dirs.sort()
        for d in dirs:
            class_names.append(full_dict[d])

        # Only first level
        break

    return class_names


# Return hard-coded class names
def class_names():
    return ['goldfish, Carassius auratus',
            'European fire salamander, Salamandra salamandra',
            'bullfrog, Rana catesbeiana',
            'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
            'American alligator, Alligator mississipiensis',
            'boa constrictor, Constrictor constrictor',
            'trilobite',
            'scorpion',
            'black widow, Latrodectus mactans',
            'tarantula',
            'centipede',
            'goose',
            'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
            'jellyfish',
            'brain coral',
            'snail',
            'slug',
            'sea slug, nudibranch',
            'American lobster, Northern lobster, Maine lobster, Homarus americanus',
            'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
            'black stork, Ciconia nigra',
            'king penguin, Aptenodytes patagonica',
            'albatross, mollymawk',
            'dugong, Dugong dugon',
            'Chihuahua',
            'Yorkshire terrier',
            'golden retriever',
            'Labrador retriever',
            'German shepherd, German shepherd dog, German police dog, alsatian',
            'standard poodle',
            'tabby, tabby cat',
            'Persian cat',
            'Egyptian cat',
            'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
            'lion, king of beasts, Panthera leo',
            'brown bear, bruin, Ursus arctos',
            'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
            'fly',
            'bee',
            'grasshopper, hopper',
            'walking stick, walkingstick, stick insect',
            'cockroach, roach',
            'mantis, mantid',
            "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
            'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
            'sulphur butterfly, sulfur butterfly',
            'sea cucumber, holothurian',
            'guinea pig, Cavia cobaya',
            'hog, pig, grunter, squealer, Sus scrofa',
            'ox',
            'bison',
            'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
            'gazelle',
            'Arabian camel, dromedary, Camelus dromedarius',
            'orangutan, orang, orangutang, Pongo pygmaeus',
            'chimpanzee, chimp, Pan troglodytes',
            'baboon',
            'African elephant, Loxodonta africana',
            'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
            'abacus',
            "academic gown, academic robe, judge's robe", 'altar',
            'apron',
            'backpack, back pack, knapsack, packsack, rucksack, haversack',
            'bannister, banister, balustrade, balusters, handrail',
            'barbershop',
            'barn',
            'barrel, cask',
            'basketball',
            'bathtub, bathing tub, bath, tub',
            'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
            'beacon, lighthouse, beacon light, pharos',
            'beaker',
            'beer bottle',
            'bikini, two-piece',
            'binoculars, field glasses, opera glasses',
            'birdhouse',
            'bow tie, bow-tie, bowtie',
            'brass, memorial tablet, plaque',
            'broom',
            'bucket, pail',
            'bullet train, bullet',
            'butcher shop, meat market',
            'candle, taper, wax light',
            'cannon',
            'cardigan',
            'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
            'CD player',
            'chain',
            'chest',
            'Christmas stocking',
            'cliff dwelling',
            'computer keyboard, keypad',
            'confectionery, confectionary, candy store',
            'convertible',
            'crane',
            'dam, dike, dyke',
            'desk',
            'dining table, board',
            'drumstick',
            'dumbbell',
            'flagpole, flagstaff',
            'fountain',
            'freight car',
            'frying pan, frypan, skillet',
            'fur coat',
            'gasmask, respirator, gas helmet',
            'go-kart',
            'gondola',
            'hourglass',
            'iPod',
            'jinrikisha, ricksha, rickshaw',
            'kimono',
            'lampshade, lamp shade',
            'lawn mower, mower',
            'lifeboat',
            'limousine, limo',
            'magnetic compass',
            'maypole',
            'military uniform',
            'miniskirt, mini',
            'moving van',
            'nail',
            'neck brace',
            'obelisk',
            'oboe, hautboy, hautbois',
            'organ, pipe organ',
            'parking meter',
            'pay-phone, pay-station',
            'picket fence, paling',
            'pill bottle',
            "plunger, plumber's helper", 'pole',
            'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
            'poncho',
            'pop bottle, soda bottle',
            "potter's wheel", 'projectile, missile',
            'punching bag, punch bag, punching ball, punchball',
            'reel',
            'refrigerator, icebox',
            'remote control, remote',
            'rocking chair, rocker',
            'rugby ball',
            'sandal',
            'school bus',
            'scoreboard',
            'sewing machine',
            'snorkel',
            'sock',
            'sombrero',
            'space heater',
            "spider web, spider's web", 'sports car, sport car',
            'steel arch bridge',
            'stopwatch, stop watch',
            'sunglasses, dark glasses, shades',
            'suspension bridge',
            'swimming trunks, bathing trunks',
            'syringe',
            'teapot',
            'teddy, teddy bear',
            'thatch, thatched roof',
            'torch',
            'tractor',
            'triumphal arch',
            'trolleybus, trolley coach, trackless trolley',
            'turnstile',
            'umbrella',
            'vestment',
            'viaduct',
            'volleyball',
            'water jug',
            'water tower',
            'wok',
            'wooden spoon',
            'comic book',
            'plate',
            'guacamole',
            'ice cream, icecream',
            'ice lolly, lolly, lollipop, popsicle',
            'pretzel',
            'mashed potato',
            'cauliflower',
            'bell pepper',
            'mushroom',
            'orange',
            'lemon',
            'banana',
            'pomegranate',
            'meat loaf, meatloaf',
            'pizza, pizza pie',
            'potpie',
            'espresso',
            'alp',
            'cliff, drop, drop-off',
            'coral reef',
            'lakeside, lakeshore',
            'seashore, coast, seacoast, sea-coast',
            'acorn']
