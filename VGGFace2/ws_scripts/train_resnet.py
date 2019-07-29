#!/usr/bin/python3

import sys
import os
from glob import glob
import numpy as np
import sklearn

import tensorflow as tf
import keras

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

from dataset_tools import load_for_training, load_for_test, dataset_size, NUM_CLASSES

batch_size = 64
epochs = 20

# learning rate schedule
initial_learning_rate = 0.0005
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 6
weight_decay = 5e-5
MULTIPLIER_FOR_OLD_LAYERS = 0.1

siz = 224

dirnm = "exp-inp%d" % (siz)
shape = (1, siz, siz, 3)

print("Setting up for %s." % dirnm)

# Carico la rete di base
# model = keras.applications.resnet50.ResNet(input_shape=(224,224,3))
from keras.applications.resnet50 import ResNet50

source_model = ResNet50(include_top=False, input_shape=(shape[1], shape[2], shape[3]), classes=4)
source_model.summary()
# source_model.load_weights('resnet_v2_0.75_96.h5')
original_layers = [x.name for x in source_model.layers]
# x = source_model.get_layer('res5c_branch2c').output  # it was working with that, will try without removing layers
x = source_model.output  # Last level of the original network, without dropout

# Modifico la rete
# x = keras.layers.GlobalAveragePooling2D()(last_layer)
# x = keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)

outS = x
outS = Dense(NUM_CLASSES, activation="softmax", name='outS')(outS)
res_model = Model(source_model.input, outS)  # Modelo solo gender
res_model_multitask = res_model
print("=======================================================================")
res_model_multitask.summary()
# plot_model(res_model_multitask, to_file=os.path.join(dirnm, 'ResNet50.png'), show_shapes=True)


# Setto i moltiplicatori del learning rate
for layer in res_model_multitask.layers:
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
from training_tools import Adam_lr_mult
from training_tools import step_decay_schedule

adam_with_lr_multipliers = Adam_lr_mult(lr=initial_learning_rate, decay=weight_decay,
                                        multipliers=learning_rate_multipliers)
res_model_multitask.compile(adam_with_lr_multipliers,
                            loss=['categorical_crossentropy'], metrics=['accuracy'])

# Preparo le callback
if not os.path.isdir(dirnm):
    os.mkdir(dirnm)
filepath = os.path.join(dirnm, "resnet.{epoch:02d}-{val_loss:.2f}.hdf5")
logdir = os.path.join(dirnm, 'tb_logs')
lr_sched = step_decay_schedule(initial_lr=initial_learning_rate, decay_factor=learning_rate_decay_factor,
                               step_size=learning_rate_decay_epochs)
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
callbacks_list = [lr_sched, checkpoint, tbCallBack]

# Addestra

if __name__ == '__main__':

    print("Training for %s is starting..." % dirnm)

    # Carica i dataset
    val_dataset = '/mnt/sdc1/acuissot/Faces_labeled/test'
    train_size = 2000000  # IMPORTANTE, SETTARE QUESTO VALORE!
    val_size = dataset_size(val_dataset)
    steps_per_epoch = train_size / batch_size
    validation_steps = val_size / batch_size

    train_generator = load_for_training('/mnt/sdc1/acuissot/newLabelDataset/train', batch_size, shape)
    val_generator = load_for_test(val_dataset, batch_size, shape)

    initial_epoch = 0
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            initial_epoch = int(sys.argv[1])
            ckpntlist = glob(os.path.join(dirnm, "resnet.%02d-*.hdf5" % initial_epoch))
        else:
            ckpntlist = [sys.argv[1]]
        res_model_multitask.load_weights(ckpntlist[0])
        from dataset_tools import load_for_pred

        data, Y_true = load_for_pred(val_dataset, batch_size, shape)
        print(Y_true.shape)
        Y_pred = res_model_multitask.predict(data, batch_size, verbose=1)
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
    res_model_multitask.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch, epochs=epochs,
                                      verbose=1, callbacks=callbacks_list,
                                      validation_data=val_generator, validation_steps=validation_steps,
                                      initial_epoch=initial_epoch
                                      )  # use_multiprocessing=True, workers=41, max_queue_size=32, )
