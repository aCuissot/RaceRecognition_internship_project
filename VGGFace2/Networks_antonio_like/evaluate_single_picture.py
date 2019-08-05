#!/usr/bin/python3

"""
Need to have saves of the model in ./trained_networks
images to test should be in ./testUrImg
"""


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
from datetime import datetime

from VGGFace2.Networks_antonio_like.dataset_tools import load_for_training, load_for_test, dataset_size, NUM_CLASSES
import VGGFace2.Networks_antonio_like.faceDetectionPart.preprocessing as prepr

print(keras.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)

# Menu Part
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
webcam = input(
    "If you want to use Webcam picture, press '1', else, if you want to use a picture on your device, press '2'\n "
    "After your choice press enter\n")
webcam = int(webcam)
webcam = webcam == 1


def getImageFromWebcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('./testUrImg/' + dt_string + '.jpg', frame)
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            cap.release()
            return './testUrImg/' + dt_string + '.jpg'


img_path = './testUrImg/bernardo.jpg'
img_extension = ["jpg", "jpeg", "png"]
if webcam:
    img_path = getImageFromWebcam()
else:
    tmp_path = input(
        "If you want to use an image, place it in 'testUrImg' dir and give its name here, else just press enter and "
        "you will use the default image\n")
    if tmp_path != "":
        img_path = './testUrImg/' + tmp_path
        if img_path.__contains__("."):
            if img_path.split(".")[-1] not in img_extension:
                print("Your image extension is not tested, if there is error, retry with a .png, .jpg or .jpeg image")
        else:
            print("The image you gave have no extension, please fix it and retry")

network = input("Please choose the network to use ('vgg16', 'vggface', 'resnet', 'nasnet', or 'mobilenet'")
checkpoint_loaded = input("Please choose the weights to load, must be in 'trained_networks' dir, default: the first "
                          "weight with a name matching ith the network you chose previously\n")
if checkpoint_loaded == "":
    folders = os.listdir("./trained_networks")
    for folder in folders:
        if folder.__contains__(network):
            checkpoint_loaded = folder
            break
if checkpoint_loaded == "":
    print("No trained model found")
else:
    print("Weights " + checkpoint_loaded + " loaded")
# checkpoint have to be in 'trained_networks' directory

num_class = 4
siz = 0
if network == "resnet" or network == "vgg16" or network == "vggface" or network == "mobilenet":
    siz = 96
elif network == "nasnet":
    siz = 331
else:
    print("Unknown network, please choose one in ['resnet', 'mobilenet', 'vgg16', 'vggface', 'nasnet']")
shape = (1, siz, siz, 3)


# max != 1 only for mobilenet!!!
def decode_pred(pred):
    pred = pred[0]
    argmax = 0
    max = 0
    for i in range(num_class):
        if pred[i] > max:
            max = pred[i]
            argmax = i
    if argmax == 0:
        return "African: " + str(max)
    elif argmax == 1:
        return "Asian: " + str(max)
    elif argmax == 2:
        return "Caucasian/Latin: " + str(max)
    else:
        return "Indian: " + str(max)


img_preprocessed, is_preprocessed = prepr.preprocessing_face_without_alignement(img_path)
if not is_preprocessed:
    print("Error during preprocessing, image does not exist or face not found")
img_preprocessed = cv2.resize(img_preprocessed, shape[1:3])
# learning rate schedule
initial_learning_rate = 0.005  # we used 0.0005 for some trainings with vggface and vgg16
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

if network == 'vggface':
    from VGGFace2.Networks_antonio_like.vggFace import VGGFace
    import VGGFace2.Networks_antonio_like.vggFace_utils as vggFacePrepro

    source_model = VGGFace(input_shape=(shape[1], shape[2], shape[3]), include_top=False, classes=4)
    original_layers = [x.name for x in source_model.layers]
    x = source_model.get_layer('conv5_3').output  # last level of the original network without dropout

    # Modify the network
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
        x = vggFacePrepro.preprocess_input(x)
        pred = model_multitask.predict(x)
        print(decode_pred(pred))

if network == 'mobilenet':
    from VGGFace2.Networks_antonio_like.mobilenet_v2_keras import MobileNetv2
    import keras.applications.mobilenet as mobilenet

    mul = 0.75
    source_model = MobileNetv2((shape[1], shape[2], shape[3]), 1001, mul)
    original_layers = [x.name for x in source_model.layers]
    x = source_model.get_layer('reshape_1').output  # last level of the original network without dropout

    # Modify the network
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
        x = mobilenet.preprocess_input(x)
        pred = model_multitask.predict(x)
        print(decode_pred(pred))

# Nasnet is not Ready (no train available => no test)

if network == 'nasnet':
    from keras.applications import NASNetLarge
    import keras.applications.nasnet as nasnet

    source_model = NASNetLarge(input_shape=(shape[1], shape[2], shape[3]), include_top=False, classes=4)
    original_layers = [x.name for x in source_model.layers]
    x = source_model.get_layer('normal_concat_18').output  # last level of the original network without dropout

    # Modify the network
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
        x = nasnet.preprocess_input(x)
        pred = model_multitask.predict(x)
        print(decode_pred(pred))
