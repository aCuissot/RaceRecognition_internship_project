# coding: utf-8

import cv2
import os
import glob
import tensorflow
from multiprocessing import freeze_support

# from sklearn.utils import shuffle
import numpy as np
import csv
import pandas as pd

import keras
from keras.preprocessing import image

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

print("IMPORT OK")

# CONFIGURATION

train_size = 675358
test_size = 24187
val_size = 0
# val_size = 216374

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
num_classes = 0

# choose the images size
img_size_row = 224
img_size_column = 224
global IMAGE_SIZE
IMAGE_SIZE = (img_size_row, img_size_column)

# choose the batch size
batch_size = 64
steps_per_epoch = train_size / batch_size
# validation_steps = val_size / batch_size

# choose the epochs
epochs = 100

# choose the learning rate
learning_rate_di_base = 0.001
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 10

# choose the weight decay
weight_decay = 5e-5

print("CONFIGURATION OK")

csvBBGroundTruthTrainIds = pd.read_csv("../Data/bb_landmark/loose_bb_test.csv").loc[:, "NAME_ID"]
csvBBGroundTruthTrain = pd.read_csv("../Data/bb_landmark/loose_bb_test.csv").set_index("NAME_ID")
Path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\"


def getPrimarySquareSize(shape, bb):
    maxBB = max(bb[2], bb[3])
    if maxBB > shape[0] or maxBB > shape[1]:
        return min(shape[0], shape[1])
    else:
        return maxBB


# Here we crop the face in a square image with a padding of 20% if it's possible

def cropFace(img, imgId):
    shape = img.shape
    bb = csvBBGroundTruthTrain.loc[imgId, :]
    primarySquareSize = getPrimarySquareSize(shape, bb)
    topleft = [int(bb[0] - ((primarySquareSize - bb[2]) / 2) - (0.2 * primarySquareSize)),
               int(bb[1] - ((primarySquareSize - bb[3]) / 2) - (0.2 * primarySquareSize))]
    botright = [int(bb[0] + primarySquareSize - ((primarySquareSize - bb[2]) / 2) + (0.2 * primarySquareSize)),
                int(bb[1] + primarySquareSize - ((primarySquareSize - bb[3]) / 2) + (0.2 * primarySquareSize))]

    topleft[0] = max(topleft[0], 0)
    topleft[1] = max(topleft[1], 0)
    botright[0] = min(botright[0], shape[0])
    botright[1] = min(botright[1], shape[1])

    return img[topleft[1]:botright[1], topleft[0]:botright[0]]


def preprocessing(img_name, img_path):
    targetSize = (224, 224)  # size for mobilenet
    print(img_path)
    img = cv2.imread(img_path)
    img_id = img_path.split("\\")[-2] + "/" + img_name.split("\\")[-1]
    img = cropFace(img, img_id)
    img = cv2.resize(img, targetSize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = keras.applications.mobilenet.preprocess_input(x)

    return x
    # x shape = (1, 224, 224, 3)....


def data_generator(csvpath):
    labs = []
    with open(csvpath, newline='') as csvfile:
        labels = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in labels:
            labs.append(row)

    p = np.array(labs)
    np.random.shuffle(p)

    for row in p:
        path = row[0]

        ethnicity = np.array(int(row[1]))

        yield (path, ethnicity)


def data_generator_prepr(csvpath, size, train):
    subpath = ""
    if train:
        subpath = "train"
    else:
        subpath = "test"
    while 1:
        data = data_generator(csvpath)
        img_list = []
        ethnicity_list = []
        for i in data:

            img = i[0]

            img = img[int(ny):int(ny + nr)][int(nx):int(nx + nr)]

            pic_path = Path + subpath + "\\" + i[0]
            picture = preprocessing(pic_path.split(".")[0], pic_path)

            picture = picture.astype(np.float32)
            # image /= 255
            # do stuff for preprocessing img
            img_list.append(picture)
            ethnicity_list.append(i[1])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(ethnicity_list, num_classes=num_classes)
                gender = np.array(gender_categorical)
                img_list = []
                ethnicity_list = []
                yield (im_p, gender)


train_generator = data_generator_prepr('../Data/labels/homogeneousCsvTrain.csv', batch_size, True)
test_generator = data_generator_prepr('../Data/labels/homogeneousCsvTest.csv', batch_size, False)
# val_generator = data_generator_prepr('../csvVggDataset2/valCompleto.csv', 1)  # to delete?

model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3))
# model.summary()


last_layer = model.get_layer('conv_pw_13_relu').output
x = keras.layers.GlobalAveragePooling2D()(last_layer)
x = keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
x = keras.layers.Dropout(1e-3, name='dropout')(x)
x = keras.layers.Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(x)
x = keras.layers.Activation('softmax', name='act_softmax')(x)
x = keras.layers.Reshape((num_classes,), name='reshape_2')(x)

mobile_model = Model(model.input, x)
# mobile_model.summary()

i = 0
for layer in mobile_model.layers:
    layer.trainable = True
    i = i + 1

print("Layer number: " + str(i))

learning_rate_multipliers = {'conv1': 0.01, 'conv1_bn': 0.01, 'conv_dw_1': 0.01, 'conv_dw_1_bn': 0.01,
                             'conv_pw_1': 0.01, 'conv_pw_1_bn': 0.01, 'conv_dw_2': 0.01, 'conv_dw_2_bn': 0.01,
                             'conv_pw_2': 0.01, 'conv_pw_2_bn': 0.01, 'conv_dw_3': 0.01, 'conv_dw_3_bn': 0.01,
                             'conv_pw_3': 0.01, 'conv_pw_3_bn': 0.01, 'conv_dw_4': 0.01, 'conv_dw_4_bn': 0.01,
                             'conv_pw_4 ': 0.01, 'conv_pw_4_bn': 0.01, 'conv_dw_5': 0.01, 'conv_dw_5_bn': 0.01,
                             'conv_pw_5 ': 0.01, 'conv_pw_5_bn': 0.01, 'conv_dw_6': 0.01, 'conv_dw_6_bn': 0.01,
                             'conv_pw_6 ': 0.01, 'conv_pw_6_bn': 0.01, 'conv_dw_7': 0.01, 'conv_dw_7_bn': 0.01,
                             'conv_pw_7 ': 0.01, 'conv_pw_7_bn': 0.01, 'conv_dw_8': 0.01, 'conv_dw_8_bn': 0.01,
                             'conv_pw_8 ': 0.01, 'conv_pw_8_bn': 0.01, 'conv_dw_9': 0.01, 'conv_dw_9_bn': 0.01,
                             'conv_pw_9 ': 0.01, 'conv_pw_9_bn': 0.01, 'conv_dw_10': 0.01, 'conv_dw_10_bn': 0.01,
                             'conv_pw_10 ': 0.01, 'conv_pw_10_bn': 0.01, 'conv_dw_11': 0.01, 'conv_dw_11_bn': 0.01,
                             'conv_pw_11 ': 0.01, 'conv_pw_11_bn': 0.01, 'conv_dw_12': 0.01, 'conv_dw_12_bn': 0.01,
                             'conv_pw_12 ': 0.01, 'conv_pw_12_bn': 0.01, 'conv_dw_13': 0.01, 'conv_dw_13_bn': 0.01,
                             'conv_pw_13 ': 0.01, 'conv_pw_13_bn': 0.01, 'conv_preds': 1}

from keras.legacy import interfaces
from keras import backend as K
from keras.optimizers import Optimizer


# class Adam_lr_mult(Optimizer):
#     """Adam optimizer.
#     Adam optimizer, with learning rate multipliers built on Keras implementation
#     # Arguments
#         lr: float >= 0. Learning rate.
#         beta_1: float, 0 < beta < 1. Generally close to 1.
#         beta_2: float, 0 < beta < 1. Generally close to 1.
#         epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
#         decay: float >= 0. Learning rate decay over each update.
#         amsgrad: boolean. Whether to apply the AMSGrad variant of this
#             algorithm from the paper "On the Convergence of Adam and
#             Beyond".
#     # References
#         - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
#         - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
#
#     AUTHOR: Erik Brorson
#     """
#
#     def __init__(self, lr=learning_rate_di_base, beta_1=0.9, beta_2=0.999,
#                  epsilon=None, decay=weight_decay, amsgrad=False,
#                  multipliers=None, debug_verbose=False, **kwargs):
#         super(Adam_lr_mult, self).__init__(**kwargs)
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.lr = K.variable(lr, name='lr')
#             self.beta_1 = K.variable(beta_1, name='beta_1')
#             self.beta_2 = K.variable(beta_2, name='beta_2')
#             self.decay = K.variable(decay, name='decay')
#         if epsilon is None:
#             epsilon = K.epsilon()
#         self.epsilon = epsilon
#         self.initial_decay = decay
#         self.amsgrad = amsgrad
#         self.multipliers = multipliers
#         self.debug_verbose = debug_verbose
#
#     @interfaces.legacy_get_updates_support
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]
#
#         lr = self.lr
#         if self.initial_decay > 0:
#             lr *= (1. / (1. + self.decay * K.cast(self.iterations,
#                                                   K.dtype(self.decay))))
#
#         t = K.cast(self.iterations, K.floatx()) + 1
#         lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
#                      (1. - K.pow(self.beta_1, t)))
#
#         ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
#         vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
#         if self.amsgrad:
#             vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
#         else:
#             vhats = [K.zeros(1) for _ in params]
#         self.weights = [self.iterations] + ms + vs + vhats
#
#         for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
#
#             # Learning rate multipliers
#             if self.multipliers:
#                 multiplier = [mult for mult in self.multipliers if mult in p.name]
#             else:
#                 multiplier = None
#             if multiplier:
#                 new_lr_t = lr_t * self.multipliers[multiplier[0]]
#                 if self.debug_verbose:
#                     print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
#                     print(K.get_value(new_lr_t))
#             else:
#                 new_lr_t = lr_t
#                 if self.debug_verbose:
#                     print('No change in learning rate {}'.format(p.name))
#                     print(K.get_value(new_lr_t))
#             m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
#             v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
#             if self.amsgrad:
#                 vhat_t = K.maximum(vhat, v_t)
#                 p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
#                 self.updates.append(K.update(vhat, vhat_t))
#             else:
#                 p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
#
#             self.updates.append(K.update(m, m_t))
#             self.updates.append(K.update(v, v_t))
#             new_p = p_t
#
#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)
#
#             self.updates.append(K.update(p, new_p))
#         return self.updates
#
#     def get_config(self):
#         config = {'lr': float(K.get_value(self.lr)),
#                   'beta_1': float(K.get_value(self.beta_1)),
#                   'beta_2': float(K.get_value(self.beta_2)),
#                   'decay': float(K.get_value(self.decay)),
#                   'epsilon': self.epsilon,
#                   'amsgrad': self.amsgrad,
#                   'multipliers': self.multipliers}
#         base_config = super(Adam_lr_mult, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def step_decay_schedule(initial_lr=learning_rate_di_base, decay_factor=learning_rate_decay_factor,
                        step_size=learning_rate_decay_epochs):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule, verbose=1)


def main():
    # adam_with_lr_multipliers = Adam_lr_mult(multipliers=learning_rate_multipliers)

    mobile_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TRAIN

    # callbacks
    filepath = "mobile5.best.hdf5"
    lr_sched = step_decay_schedule()
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [EarlyStopping(monitor='val_acc',
                                    patience=7,
                                    verbose=1,
                                    min_delta=1e-4,
                                    mode='max'),
                      lr_sched,
                      checkpoint]

    print("jaaj")
    index = 0
    for i in test_generator:
        if i[0].shape != (64, 1, 224, 224, 3):
            print("1 PAS OK,", i[0].shape)
        if i[1].shape != (64, 5):
            print("2 PAS OK,", i[1].shape)
        print(index)
        index += 1
    print("end")

    mobile_model.fit_generator(test_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                               callbacks=callbacks_list)

    scores = mobile_model.evaluate_generator(test_generator, steps=test_size, verbose=1)

    print("%s: %.2f%%" % (mobile_model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    freeze_support()
    main()
