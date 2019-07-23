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
from keras_vggface.vggface import VGGFace
from keras import optimizers
from keras.constraints import maxnorm

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import pickle

print("IMPORT OK")

# CONFIGURATION

# img_media=mt.image.imread("immagine_media.jpg")
img_media = cv2.imread('immagine_mediaCroppedCV2.jpg')
img_media = img_media.astype(np.float32)
img_media /= 255

# total ...-> 70%train 15%val 15%test
train_size = 2000000
test_size = 121594
val_size = 216374

dataset_size = train_size + val_size + test_size

# Parameters for face crop
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
dropout_rate = 0.5

# choose the fully connected neurons
fc_neurons = 512

# choose the batch size
batch_size = 64
steps_per_epoch = train_size / batch_size
validation_steps = val_size / batch_size

# choose the epochs
epochs = 100

# choose the learning rate
learning_rate = 0.001
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 5

# choose the momentum
momentum = 0.9

# choose the weight decay
weight_decay = 5 * 1e-4

print("CONFIGURATION OK")


def data_generator(csvpath):
    labs = []
    with open(csvpath, newline='') as csvfile:
        labels = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in labels:
            labs.append(row)

    p = np.array(labs)
    np.random.shuffle(p)

    for row in p:
        path = row[2]
        image = cv2.imread(path)

        im = np.array(image)
        gender = np.array(int(row[0]))

        yield (im, gender)


def data_generator_prepr(csvpath, size):
    while 1:
        data = data_generator(csvpath)
        img_list = []
        gender_list = []
        for i in data:

            # cropping face only
            img = i[0]

            img = img[ny:ny + nr, nx:nx + nr]

            image = cv2.resize(img, IMAGE_SIZE, 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255

            # normalization
            image -= img_media

            img_list.append(image)
            gender_list.append(i[1])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(gender_list, num_classes=num_classes)
                gender = np.array(gender_categorical)
                img_list = []
                gender_list = []
                yield (im_p, gender)


train_generator = data_generator_prepr('trainCompleto.csv', batch_size)
val_generator = data_generator_prepr('valCompleto.csv', batch_size)
test_generator = data_generator_prepr('testCompleto.csv', 1)

vgg_model = VGGFace(include_top=False, input_shape=(img_size_column, img_size_row, 3))
# Preparation for fine tuning
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(fc_neurons, activation='relu', name='fc6', kernel_constraint=maxnorm(3))(x)
x = Dense(fc_neurons, activation='relu', name='fc7', kernel_constraint=maxnorm(3))(x)
x = Dropout(dropout_rate)(x)
out = Dense(num_classes, activation='softmax', name='fc8')(x)

custom_vgg_model = Model(vgg_model.input, out)

for layer in custom_vgg_model.layers:
    layer.trainable = False
custom_vgg_model.get_layer('fc6').trainable = True
custom_vgg_model.get_layer('fc7').trainable = True
custom_vgg_model.get_layer('fc8').trainable = True

custom_vgg_model.summary()


def step_decay_schedule(initial_lr=learning_rate, decay_factor=learning_rate_decay_factor,
                        step_size=learning_rate_decay_epochs):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule, verbose=1)


custom_vgg_model.compile(optimizers.SGD(lr=learning_rate, momentum=momentum, decay=weight_decay),
                         loss='categorical_crossentropy', metrics=['accuracy'])

# callbacks
filepath = "vggdef3.best.hdf5"
lr_sched = step_decay_schedule()
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [EarlyStopping(monitor='val_acc',
                                patience=7,
                                verbose=1,
                                min_delta=1e-4,
                                mode='max'),
                  lr_sched,
                  checkpoint]

custom_vgg_model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                               callbacks=callbacks_list,
                               validation_data=val_generator, validation_steps=validation_steps,
                               use_multiprocessing=True)

scores = custom_vgg_model.evaluate_generator(test_generator, steps=test_size, use_multiprocessing=True, verbose=1)

print("%s: %.2f%%" % (custom_vgg_model.metrics_names[1], scores[1] * 100))
