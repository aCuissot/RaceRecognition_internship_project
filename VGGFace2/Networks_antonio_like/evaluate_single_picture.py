#!/usr/bin/python3

import sys
import os
from glob import glob
import numpy as np
import sklearn

import tensorflow as tf
import keras
from keras.preprocessing import image

print(keras.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)

from keras.engine import Model
from keras.layers import Flatten, Dense, Dropout, Input, Reshape, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras import backend as K

from VGGFace2.Networks_antonio_like.dataset_tools import load_for_training, load_for_test, dataset_size, NUM_CLASSES

img_path = './testUrImg/jaaj.png'

batch_size = 64
epochs = 64

# learning rate schedule
initial_learning_rate = 0.005
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 6
weight_decay = 5e-5
MULTIPLIER_FOR_OLD_LAYERS = 0.1

siz = 96

dirnm = "trained_networks"
shape = (1, siz, siz, 3)

print("Setting up for %s." % dirnm)

# Load the basic network
# model = keras.applications.mobilenet.MobileNet(input_shape=(224,224,3))
from keras.applications.resnet50 import ResNet50, preprocess_input

source_model = ResNet50(input_shape=(shape[1], shape[2], shape[3]), include_top=False, classes=4)
original_layers = [x.name for x in source_model.layers]
x = source_model.get_layer('res5c_branch2c').output  # last level of the original network without dropout

# Modify the network
# x = keras.layers.GlobalAveragePooling2D()(last_layer)
# x = keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
x = keras.layers.Dropout(0.5, name='dropout')(x)
x = keras.layers.Flatten()(x)
outS = x
outS = Dense(NUM_CLASSES, activation="softmax", name='outS')(outS)
res_model = Model(source_model.input, outS)
res_model_multitask = res_model
res_model_multitask.summary()
# plot_model(vgg_model_multitask, to_file=os.path.join(dirnm, 'ResNet50.png'), show_shapes=True)


# based on the learning rate multipliers
for layer in res_model_multitask.layers:
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
from VGGFace2.Networks_antonio_like.training_tools import step_decay_schedule

adam_with_lr_multipliers = Adam_lr_mult(lr=initial_learning_rate, decay=weight_decay,
                                        multipliers=learning_rate_multipliers)
res_model_multitask.compile(adam_with_lr_multipliers,
                            loss=['categorical_crossentropy'], metrics=['accuracy'])

# Preparing callback
if not os.path.isdir(dirnm):
    os.mkdir(dirnm)
filepath = os.path.join(dirnm, "resnet.{epoch:02d}-{val_loss:.2f}.hdf5")
logdir = os.path.join(dirnm, 'tb_logs')
lr_sched = step_decay_schedule(initial_lr=initial_learning_rate, decay_factor=learning_rate_decay_factor,
                               step_size=learning_rate_decay_epochs)
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
callbacks_list = [lr_sched, checkpoint, tbCallBack]

# Evaluate

if __name__ == '__main__':

    print("Evaluation for %s is starting..." % dirnm)

    ckpntlist = ['resnet.16-0.22.hdf5']
    # ckpntlist = [sys.argv[1]]
    res_model_multitask.load_weights(dirnm + '/' + ckpntlist[0])

    """ data, Y_true = load_for_pred(val_dataset, batch_size, shape)
    print(Y_true.shape)
    Y_pred = vgg_model_multitask.predict(data, batch_size, verbose=1) """
    img = image.load_img(img_path, target_size=(96, 96))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # pred = res_model_multitask.predict(x)
    pred = keras.utils.to_categorical(Y_true, num_classes=NUM_CLASSES)
    print(pred)

