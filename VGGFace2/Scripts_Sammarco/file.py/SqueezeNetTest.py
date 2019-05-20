# coding: utf-8


import cv2
import os
import glob
import tensorflow

import numpy as np
import csv

import keras

print(keras.__version__)

import sys
from keras.engine import Model
from keras.layers import Flatten, Dense, Dropout, Input

from keras import optimizers
from keras.constraints import maxnorm

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import pickle

from keras_squeezenet import SqueezeNet
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

print("IMPORT OK")

# CONFIGURATION


# total ... ~> 70%train 15%val 15%test
train_size = 80000
test_size = 20000
val_size = 19997

dataset_size = train_size + val_size + test_size

# parameters for face crop
x = 51
yy = 51
w = 154
h = 154
r = max(w, h) / 2
centerx = x + w / 2
centery = yy + h / 2
nx = int(centerx - r)
ny = int(centery - r)
nr = int(r * 2)

# num of classes to be trained
num_classes = 2

# choose the images size
img_size_row = 224
img_size_column = 224
global IMAGE_SIZE
IMAGE_SIZE = (img_size_row, img_size_column)

# choose the dropout rate
dropout_rate = 0.8

# choose the batch size
batch_size = 64
steps_per_epoch = train_size / batch_size
validation_steps = val_size / batch_size

# choose the epochs
epochs = 2000

# choose the learning rate
learning_rate = 0.0001
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 10

# choose the weight decay
weight_decay = 5e-5

print("CONFIGURAZIONE OK")


# In[3]:


def data_generator(csvpath):
    labs = []
    with open(csvpath, newline='') as csvfile:
        labels = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in labels:
            labs.append(row)

    p = np.array(labs)
    np.random.shuffle(p)

    for row in p:
        path = 'C:/Users/Enrico/Desktop/tirocinio/ALL/TestDataset/' + row[2]
        image = cv2.imread(path)

        im = np.array(image)
        gender = np.array(int(row[0]))

        yield (im, gender)


def data_generator_prepr_crop(csvpath, size):
    while 1:
        data = data_generator(csvpath)
        img_list = []
        gender_list = []
        for i in data:

            img = i[0]

            # crop only face
            if img.shape == (256, 256, 3):  # solo per feret

                img = img[ny:ny + nr, nx:nx + nr]
                image = cv2.resize(img, IMAGE_SIZE, 0, 0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image /= 255

                img_list.append(image)
                gender_list.append(i[1])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(gender_list, num_classes=num_classes)
                gender = np.array(gender_categorical)
                img_list = []
                gender_list = []
                yield (im_p, gender)


def data_generator_prepr_no_crop(csvpath, size):
    while 1:
        data = data_generator(csvpath)
        img_list = []
        gender_list = []
        for i in data:

            img = i[0]

            image = cv2.resize(img, IMAGE_SIZE, 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255

            img_list.append(image)
            gender_list.append(i[1])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(gender_list, num_classes=num_classes)
                gender = np.array(gender_categorical)
                img_list = []
                gender_list = []
                yield (im_p, gender)


squeeze_model = SqueezeNet(weights=None, input_shape=(img_size_column, img_size_row, 3), classes=num_classes)

i = 0
for layer in squeeze_model.layers:
    layer.trainable = True
    i = i + 1

print("Layer number: " + str(i))


def step_decay_schedule(initial_lr=learning_rate, decay_factor=learning_rate_decay_factor,
                        step_size=learning_rate_decay_epochs):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule, verbose=1)


squeeze_model.compile(optimizers.RMSprop(lr=learning_rate, decay=weight_decay),
                      loss='categorical_crossentropy', metrics=['accuracy'])

squeeze_model.summary()

squeeze_model.load_weights("C:/Users/Enrico/Desktop/tirocinio/ALL/pesi/squeezedef2.best.hdf5")

test_size = 14755
test = data_generator_prepr_crop('C:/Users/Enrico/Desktop/tirocinio/ALL/TestDataset/lfw_cropped.csv', 1)
score = squeeze_model.evaluate_generator(test, steps=test_size, verbose=1)
print("LFW acc: ")
print("%s: %.2f%%" % (squeeze_model.metrics_names[1], score[1] * 100))
print("\n\n")

test_size = 760
test = data_generator_prepr_crop('C:/Users/Enrico/Desktop/tirocinio/ALL/TestDataset/feret_cropped_enr.csv', 1)
score = squeeze_model.evaluate_generator(test, steps=test_size, use_multiprocessing=True, verbose=1)
print("FERET acc: ")
print("%s: %.2f%%" % (squeeze_model.metrics_names[1], score[1] * 100))
print("\n\n")

test_size = 404
test = data_generator_prepr_no_crop('C:/Users/Enrico/Desktop/tirocinio/ALL/TestDataset/unisa.csv', 1)
score = squeeze_model.evaluate_generator(test, steps=test_size, use_multiprocessing=True, verbose=1)
print("UNISA acc: ")
print("%s: %.2f%%" % (squeeze_model.metrics_names[1], score[1] * 100))
print("\n\n")

test_size = 5402
test = data_generator_prepr_no_crop('C:/Users/Enrico/Desktop/tirocinio/ALL/TestDataset/unisa_private.csv', 1)
score = squeeze_model.evaluate_generator(test, steps=test_size, use_multiprocessing=True, verbose=1)
print("UNISA acc: ")
print("%s: %.2f%%" % (squeeze_model.metrics_names[1], score[1] * 100))
print("\n\n")
