#!/usr/bin/python3

import sys
import os
from glob import glob

import cv2
import numpy as np
import sklearn

import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.engine import Model
from keras.layers import Flatten, Dense, Dropout, Input, Reshape, Convolution2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras import backend as K

from VGGFace2.Networks_antonio_like.dataset_tools import load_for_training, load_for_test, dataset_size, NUM_CLASSES
import VGGFace2.Networks_antonio_like.faceDetectionPart.preprocessing as prepr

print(keras.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)

img_path = './testUrImg/mariolle.png'
network = "vgg16"
# checkpoint have to be in 'trained_networks' directory
checkpoint_loaded = 'vgg16.17-0.21.hdf5'

num_class=  4
siz = 0
if network == "resnet" or network == "vgg16" or network == "vggface" or network == "mobilenet":
    siz = 96
elif network == "nasnet":
    siz = 331
else:
    print("unknown network, please choose one in ['resnet', 'mobilenet', 'vgg16', 'vggface', 'nasnet']")

shape = (1, siz, siz, 3)


def decode_pred(pred):
    pred = pred[0]
    argmax = 0
    max = 0
    for i in range(num_class):
        if pred[i]>max:
            max = pred[i]
            argmax = i
    if argmax == 0:
        return ("African: " + str(100 * max) + "%")
    elif argmax == 1:
        return("Asian: " + str(100 * max) + "%")
    elif argmax == 2:
        return("Caucasian/Latin: " + str(100 * max) + "%")
    else:
        return("Indian: " + str(100 * max) + "%")


img_preprocessed, is_preprocessed = prepr.preprocessing_face_without_alignement(img_path)
img_preprocessed = cv2.resize(img_preprocessed, shape[1:3])
# learning rate schedule
initial_learning_rate = 0.005
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 6
weight_decay = 5e-5
MULTIPLIER_FOR_OLD_LAYERS = 0.1

dirnm = "trained_networks"

print("Setting up for %s." % dirnm)

# Load the basic network
if network == 'resnet':
    import keras.applications.resnet50 as resnet50

    source_model = resnet50.ResNet50(input_shape=(shape[1], shape[2], shape[3]), include_top=False, classes=4)
    original_layers = [x.name for x in source_model.layers]
    x = source_model.get_layer('res5c_branch2c').output  # last level of the original network without dropout

    # Modify the network
    # x = keras.layers.GlobalAveragePooling2D()(last_layer)
    # x = keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
    x = keras.layers.Dropout(0.5, name='dropout')(x)
    x = keras.layers.Flatten()(x)
    outS = x
    outS = Dense(NUM_CLASSES, activation="softmax", name='outS')(outS)
    model = Model(source_model.input, outS)
    model_multitask = model

    # based on the learning rate multipliers
    for layer in model_multitask.layers:
        layer.trainable = True
    learning_rate_multipliers = {}
    for layer_name in original_layers:
        learning_rate_multipliers[layer_name] = MULTIPLIER_FOR_OLD_LAYERS
    # added levels will have lr_multiplier = 1
    new_layers = [x.name for x in source_model.layers if x.name not in original_layers]
    for layer_name in new_layers:
        learning_rate_multipliers[layer_name] = 1

    # Prepare optimization with the lr_decay
    from VGGFace2.Networks_antonio_like.training_tools import Adam_lr_mult

    adam_with_lr_multipliers = Adam_lr_mult(lr=initial_learning_rate, decay=weight_decay,
                                            multipliers=learning_rate_multipliers)
    model_multitask.compile(adam_with_lr_multipliers,
                            loss=['categorical_crossentropy'], metrics=['accuracy'])

    # Evaluate
    if is_preprocessed:
        print("Evaluation for %s is starting..." % dirnm)

        model_multitask.load_weights(dirnm + '/' + checkpoint_loaded)

        x = image.img_to_array(img_preprocessed)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        pred = model_multitask.predict(x)
        print(pred)

if network == 'vgg16':
    import keras.applications.vgg16 as vgg16

    source_model = vgg16.VGG16(input_shape=(shape[1], shape[2], shape[3]), include_top=False, classes=4)
    original_layers = [x.name for x in source_model.layers]
    x = source_model.get_layer('block5_conv3').output  # last level of the original network without dropout

    # Modify the network
    # x = keras.layers.GlobalAveragePooling2D()(last_layer)
    # x = keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
    x = keras.layers.Dropout(0.5, name='dropout')(x)
    x = keras.layers.Flatten()(x)
    outS = x
    outS = Dense(NUM_CLASSES, activation="softmax", name='outS')(outS)
    model = Model(source_model.input, outS)
    model_multitask = model

    # based on the learning rate multipliers
    for layer in model_multitask.layers:
        layer.trainable = True
    learning_rate_multipliers = {}
    for layer_name in original_layers:
        learning_rate_multipliers[layer_name] = MULTIPLIER_FOR_OLD_LAYERS
    # added levels will have lr_multiplier = 1
    new_layers = [x.name for x in source_model.layers if x.name not in original_layers]
    for layer_name in new_layers:
        learning_rate_multipliers[layer_name] = 1

    # Prepare optimization with the lr_decay
    from VGGFace2.Networks_antonio_like.training_tools import Adam_lr_mult

    adam_with_lr_multipliers = Adam_lr_mult(lr=initial_learning_rate, decay=weight_decay,
                                            multipliers=learning_rate_multipliers)
    model_multitask.compile(adam_with_lr_multipliers,
                            loss=['categorical_crossentropy'], metrics=['accuracy'])

    # Evaluate
    if is_preprocessed:
        print("Evaluation for %s is starting..." % dirnm)

        model_multitask.load_weights(dirnm + '/' + checkpoint_loaded)

        x = image.img_to_array(img_preprocessed)
        x = np.expand_dims(x, axis=0)
        x = vgg16.preprocess_input(x)
        pred = model_multitask.predict(x)
        print(decode_pred(pred))
