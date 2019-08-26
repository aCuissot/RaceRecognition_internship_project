#!/usr/bin/python3
import sys
import os
import timeit
from glob import glob
import numpy as np
import sklearn

import tensorflow as tf
import keras

from dataset_tools import load_for_pred
from mobilenet_v2_keras import MobileNetv2, relu6
from training_tools import Adam_lr_mult, step_decay_schedule

print(keras.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)

from keras.engine import Model
from keras.layers import Flatten, Dense, Dropout, Input, Reshape, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras import backend as K

from dataset_tools import load_for_training, load_for_test, dataset_size, NUM_CLASSES

batch_size = 64
epochs = 1

# learning rate schedule
initial_learning_rate = 0.005
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 6
weight_decay = 5e-5
MULTIPLIER_FOR_OLD_LAYERS = 0.1

siz = 224
mul = 0.75

dirnm = "exp-inp%d-mul%s" % (siz, str(mul))
shape = (1, siz, siz, 3)

print("Setting up for %s." % dirnm)

# Carico la rete di base
# model = keras.applications.mobilenet.MobileNet(input_shape=(224,224,3))

source_model = MobileNetv2((shape[1], shape[2], shape[3]), 1001, mul)
source_model.load_weights('mobilenet_v2_0.75_96.h5')
original_layers = [x.name for x in source_model.layers]
x = source_model.get_layer('reshape_1').output  # Ultimo livello della rete originale, senza dropout

# Modifico la rete
# x = keras.layers.GlobalAveragePooling2D()(last_layer)
# x = keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
x = keras.layers.Dropout(0.5, name='dropout')(x)
x = keras.layers.Flatten()(x)
outS = x
outS = Dense(NUM_CLASSES, activation="softmax", name='outS')(outS)
mobile_model = Model(source_model.input, outS)  # Modelo solo gender
mobile_model_multitask = mobile_model
mobile_model_multitask.summary()
# plot_model(mobile_model_multitask, to_file=os.path.join(dirnm, 'MobileNetv2.png'), show_shapes=True)


# Setto i moltiplicatori del learning rate
for layer in mobile_model_multitask.layers:
    layer.trainable = True
learning_rate_multipliers = {}
for layer_name in original_layers:
    learning_rate_multipliers[layer_name] = MULTIPLIER_FOR_OLD_LAYERS
# I livelli aggiunti avranno lr multiplier = 1
new_layers = [x.name for x in source_model.layers if x.name not in original_layers]
for layer_name in new_layers:
    learning_rate_multipliers[layer_name] = 1

# Ottimizza gender e age
# Preparo l'ottimizzatore con il lr decay


adam_with_lr_multipliers = Adam_lr_mult(lr=initial_learning_rate, decay=weight_decay,
                                        multipliers=learning_rate_multipliers)
mobile_model_multitask.compile(adam_with_lr_multipliers,
                               loss=['categorical_crossentropy'], metrics=['accuracy'])

# Preparo le callback
if not os.path.isdir(dirnm):
    os.mkdir(dirnm)
filepath = os.path.join(dirnm, "mobile.{epoch:02d}-{val_loss:.2f}.hdf5")
logdir = os.path.join(dirnm, 'tb_logs')
"""
lr_sched = step_decay_schedule(initial_lr=initial_learning_rate, decay_factor=learning_rate_decay_factor,
                               step_size=learning_rate_decay_epochs)
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
"""
class TimeCalc(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimeCalc, self).__init__()
        self.start = 0
        self.start_batch = 0

    def on_epoch_end(self, epoch, logs=None):
        print('Epoch time:' + str(timeit.default_timer() - self.start))

    def on_epoch_begin(self, epoch, logs=None):
        self.start = timeit.default_timer()

    def on_batch_end(self, batch, logs=None):
        print('Batch time:' + str(timeit.default_timer() - self.start_batch))

    def on_batch_begin(self, batch, logs=None):
        self.start_batch = timeit.default_timer()

timecalc = TimeCalc()
#callbacks_list = [lr_sched, checkpoint, tbCallBack, timecalc]
callbacks_list = [timecalc]

# Addestra

if __name__ == '__main__':

    print("Training for %s is starting..." % dirnm)

    # Carica i dataset
    val_dataset = '../newLabelDataset/test'
    train_size = 640  # IMPORTANTE, SETTARE QUESTO VALORE!
    val_size = dataset_size(val_dataset)
    steps_per_epoch = train_size / batch_size
    validation_steps = val_size / batch_size

    train_generator = load_for_training('../newLabelDataset/train', batch_size, shape)
    val_generator = load_for_test(val_dataset, batch_size, shape)

    initial_epoch = 0
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            initial_epoch = int(sys.argv[1])
            ckpntlist = glob(os.path.join(dirnm, "mobile.%02d-*.hdf5" % initial_epoch))
        else:
            ckpntlist = [sys.argv[1]]
        mobile_model_multitask.load_weights(ckpntlist[0])

        data, Y_true = load_for_pred(val_dataset, batch_size, shape)
        print(Y_true.shape)
        Y_pred = mobile_model_multitask.predict(data, batch_size, verbose=1)
        print(Y_pred.shape)
        print(Y_true.shape)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = np.argmax(Y_true, axis=1)
        print(y_pred.shape)
        print(y_true.shape)
        print('Confusion Matrix')
        # predneg predpos
        #  [[4178  659]  < True negative
        #   [ 975 3861]] < True positive
        from sklearn.metrics import classification_report, confusion_matrix

        # y_pred = [1]*y_true.shape[0] # Per provare, dovrebbe dare recall=1
        conf = confusion_matrix(y_true, y_pred, [0, 1, 2, 3])
        print(conf)
    start_timer = timeit.default_timer()
    mobile_model_multitask.fit_generator(train_generator,
                                         steps_per_epoch=steps_per_epoch, epochs=epochs,
                                         verbose=1, callbacks=callbacks_list,
                                         validation_data=val_generator, validation_steps=validation_steps,
                                         initial_epoch=initial_epoch
                                         )
    """ mobile_model_multitask.fit_generator(train_generator,
                                steps_per_epoch=steps_per_epoch, epochs=epochs,
                                verbose=1, callbacks=callbacks_list,
                                validation_data=val_generator, validation_steps=validation_steps, initial_epoch=initial_epoch
                                )#use_multiprocessing=True, workers=41, max_queue_size=32, ) """
