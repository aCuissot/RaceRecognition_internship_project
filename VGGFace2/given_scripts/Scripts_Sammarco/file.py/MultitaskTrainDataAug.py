# coding: utf-8


import cv2
import os
import glob
import tensorflow as tf

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
import random

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.utils import np_utils

print("IMPORT OK")

train_size = 400000
test_size = 2912
val_size = 50000

dataset_size = train_size + val_size + test_size

# num of classes to be trained
num_classes = 2

# choose the images size
img_size_row = 224
img_size_column = 224
global IMAGE_SIZE
IMAGE_SIZE = (img_size_row, img_size_column)

# choose the batch size
batch_size = 64
steps_per_epoch = train_size / batch_size
validation_steps = val_size / batch_size

# choose the epochs
epochs = 100

# choose the learning rate
learning_rate_di_base = 0.001
learning_rate_decay_factor = 0.5
learning_rate_decay_epochs = 10

# choose the weight decay
weight_decay = 5e-5

print("CONFIGURATION OK")


def onlinedataaug(img):
    # if they all come out 0, no data augmentation, otherwise sum of the effects of the 1 coming out
    if random.randint(0, 1):  # gauss
        # print("gauss")
        row, col, ch = img.shape

        binom = np.random.binomial(n=500, p=0.5, size=(row, col, ch))
        binom = binom - 250

        noisy = img + binom

        for i in range(0, row):
            for j in range(0, col):
                for l in range(0, ch):
                    if noisy[i][j][l] < 0:
                        noisy[i][j][l] = 0
                    elif noisy[i][j][l] > 255:
                        noisy[i][j][l] = 255

        img = noisy.astype(np.uint8)

    if random.randint(0, 1):  # resize
        # print("resize")
        row, col, ch = img.shape
        p = random.randint(1, 2) * 2
        row = row / p
        col = col / p
        img = cv2.resize(img, (int(row), int(col)), 0, 0, cv2.INTER_LINEAR)

    # if random.randint(0,1):  #brightess  alternatamente aggiungo e sottraggo illuminazione
    # print("brightness")
    #    exposition = random.randint(1,2) #se 1 sovraespongo se 2 sottoespongo
    #    if exposition==2:
    #        param=random.randint(1,2)/10
    #        param = 0 - param

    #    else :
    #        param=random.randint(2,3)/10

    #    br= tf.image.adjust_brightness(immagine,param)
    #    with tf.Session() as sess:
    #        sess.run(tf.global_variables_initializer())
    #        immagine = sess.run(br)
    #        sess.close()

    # if random.randint(0,1): #flip
    #    #print("flip")
    #    imgflip = tf.image.flip_left_right(immagine)
    #    with tf.Session() as sess:
    #        sess.run(tf.global_variables_initializer())
    #        immagine = sess.run(imgflip)
    #        sess.close()

    if random.randint(0, 1):  # brightess: alternately I add and subtract lighting
        # print("brightness")

        exposition = random.randint(1, 2)  # 1 = overexposed, 2 = underexposed
        if exposition == 2:
            brig = img * 0.7

        else:
            brig = img * 1.4

        row, col, ch = brig.shape
        for i in range(0, row):
            for j in range(0, col):
                for k in range(0, ch):
                    if brig[i][j][k] > 255:
                        brig[i][j][k] = 255
                    elif brig[i][j][k] < 0:
                        brig[i][j][k] = 0
        img = brig.astype(np.uint8)

    if random.randint(0, 1):  # flip
        # print("flip")
        flip = np.fliplr(img)
        img = flip

    return img


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
        age = np.array(int(row[1]))

        yield (im, gender, age)


def data_generator_prepr(csvpath, size):
    while 1:
        data = data_generator(csvpath)
        img_list = []
        gender_list = []
        age_list = []
        for i in data:
            # adding same definition of ny, nx, nr than bellow, hoping that s good
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


            picture = i[0]

            picture = picture[ny:ny + nr, nx:nx + nr]

            image = cv2.resize(picture, IMAGE_SIZE, 0, 0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)
            image /= 255

            img_list.append(image)
            gender_list.append(i[1])
            age_list.append(i[2])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(gender_list, num_classes=num_classes)

                # age=age_list.copy()
                # gender=np.array(gender_categorical)
                age = np.array(age_list)
                # labels=np.array([gender,[age]])
                img_list = []
                gender_list = []
                age_list = []
                yield (im_p, [gender_categorical, age])


def data_generator_con_data_aug(csvpath, size):
    while 1:
        data = data_generator(csvpath)
        img_list = []
        gender_list = []
        age_list = []
        for i in data:

            picture = i[0]

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

            # random crop con prob + alta su 0 (gaussiana)
            sigma = 8
            mean = 0
            gauss = np.random.normal(mean, sigma, (2, 1))

            gauss = gauss.astype(int)
            if gauss[0] > 8:
                gauss[0] = 8
            elif gauss[0] < -8:
                gauss[0] = -8
            if gauss[1] > 8:
                gauss[1] = 8
            elif gauss[1] < -8:
                gauss[1] = -8

            ny = ny + gauss[0][0]
            nx = nx + gauss[1][0]

            picture = picture[ny:ny + nr, nx:nx + nr]

            image = onlinedataaug(picture)

            image = cv2.resize(image, IMAGE_SIZE, 0, 0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)
            image /= 255

            img_list.append(image)
            gender_list.append(i[1])
            age_list.append(i[2])

            if len(img_list) == size:
                im_p = np.array(img_list)
                gender_categorical = keras.utils.to_categorical(gender_list, num_classes=num_classes)

                age = np.array(age_list)
                img_list = []
                gender_list = []
                age_list = []
                yield (im_p, [gender_categorical, age])


model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3))

last_layer = model.get_layer('conv_pw_13_relu').output
x = keras.layers.GlobalAveragePooling2D()(last_layer)
x = keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
x = keras.layers.Dropout(1e-3, name='dropout')(x)
x = keras.layers.Conv2D(num_classes, (1, 1),

                        padding='same',

                        name='conv_preds')(x)
x = keras.layers.Activation('softmax', name='act_softmax')(x)
x = keras.layers.Reshape((num_classes,), name='reshape_2')(x)

mobile_model = Model(model.input, x)

dropout = mobile_model.get_layer('dropout').output
last_ConvLayer = mobile_model.get_layer('conv_preds').output
x1 = keras.layers.Activation('softmax', name='act_softmax')(last_ConvLayer)
out1 = keras.layers.Reshape((num_classes,), name='outG')(x1)
x2 = keras.layers.Flatten()(dropout)
x2 = Dense(500, name='dense_penultimoAGE')(x2)
x2 = keras.layers.Dropout(0.5, name='dropout_tra_i_dense')(x2)
out2 = Dense(1, activation="linear", name='outA')(x2)

mobile_model_multitask = Model(mobile_model.input, [out1, out2])

mobile_model_multitask.summary()

mobile_model_multitask.load_weights("mobilemultitaskdibase.best.hdf5")

i = 0
for layer in mobile_model_multitask.layers:
    layer.trainable = True
    i = i + 1

print("Numero di layer: " + str(i))

learning_rate_multipliers = {}
learning_rate_multipliers['conv1'] = 0.01
learning_rate_multipliers['conv1_bn'] = 0.01
learning_rate_multipliers['conv_dw_1'] = 0.01
learning_rate_multipliers['conv_dw_1_bn'] = 0.01
learning_rate_multipliers['conv_pw_1'] = 0.01
learning_rate_multipliers['conv_pw_1_bn'] = 0.01
learning_rate_multipliers['conv_dw_2'] = 0.01
learning_rate_multipliers['conv_dw_2_bn'] = 0.01
learning_rate_multipliers['conv_pw_2'] = 0.01
learning_rate_multipliers['conv_pw_2_bn'] = 0.01
learning_rate_multipliers['conv_dw_3'] = 0.01
learning_rate_multipliers['conv_dw_3_bn'] = 0.01
learning_rate_multipliers['conv_pw_3'] = 0.01
learning_rate_multipliers['conv_pw_3_bn'] = 0.01
learning_rate_multipliers['conv_dw_4'] = 0.01
learning_rate_multipliers['conv_dw_4_bn'] = 0.01
learning_rate_multipliers['conv_pw_4 '] = 0.01
learning_rate_multipliers['conv_pw_4_bn'] = 0.01
learning_rate_multipliers['conv_dw_5'] = 0.01
learning_rate_multipliers['conv_dw_5_bn'] = 0.01
learning_rate_multipliers['conv_pw_5 '] = 0.01
learning_rate_multipliers['conv_pw_5_bn'] = 0.01
learning_rate_multipliers['conv_dw_6'] = 0.01
learning_rate_multipliers['conv_dw_6_bn'] = 0.01
learning_rate_multipliers['conv_pw_6 '] = 0.01
learning_rate_multipliers['conv_pw_6_bn'] = 0.01
learning_rate_multipliers['conv_dw_7'] = 0.01
learning_rate_multipliers['conv_dw_7_bn'] = 0.01
learning_rate_multipliers['conv_pw_7 '] = 0.01
learning_rate_multipliers['conv_pw_7_bn'] = 0.01
learning_rate_multipliers['conv_dw_8'] = 0.01
learning_rate_multipliers['conv_dw_8_bn'] = 0.01
learning_rate_multipliers['conv_pw_8 '] = 0.01
learning_rate_multipliers['conv_pw_8_bn'] = 0.01
learning_rate_multipliers['conv_dw_9'] = 0.01
learning_rate_multipliers['conv_dw_9_bn'] = 0.01
learning_rate_multipliers['conv_pw_9 '] = 0.01
learning_rate_multipliers['conv_pw_9_bn'] = 0.01
learning_rate_multipliers['conv_dw_10'] = 0.01
learning_rate_multipliers['conv_dw_10_bn'] = 0.01
learning_rate_multipliers['conv_pw_10 '] = 0.01
learning_rate_multipliers['conv_pw_10_bn'] = 0.01
learning_rate_multipliers['conv_dw_11'] = 0.01
learning_rate_multipliers['conv_dw_11_bn'] = 0.01
learning_rate_multipliers['conv_pw_11 '] = 0.01
learning_rate_multipliers['conv_pw_11_bn'] = 0.01
learning_rate_multipliers['conv_dw_12'] = 0.01
learning_rate_multipliers['conv_dw_12_bn'] = 0.01
learning_rate_multipliers['conv_pw_12 '] = 0.01
learning_rate_multipliers['conv_pw_12_bn'] = 0.01
learning_rate_multipliers['conv_dw_13'] = 0.01
learning_rate_multipliers['conv_dw_13_bn'] = 0.01
learning_rate_multipliers['conv_pw_13 '] = 0.01
learning_rate_multipliers['conv_pw_13_bn'] = 0.01
learning_rate_multipliers['conv_preds'] = 1  # ultimo col lr normale
learning_rate_multipliers['flatten_2'] = 1
learning_rate_multipliers['dense_penultimoAGE'] = 1
learning_rate_multipliers['act_softmax'] = 1
learning_rate_multipliers['dropout_tra_i_dense'] = 1
learning_rate_multipliers['outG'] = 1
learning_rate_multipliers['outA'] = 1

from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer


class Adam_lr_mult(Optimizer):
    """Adam optimizer.
    Adam optimizer, with learning rate multipliers built on Keras implementation
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        
    AUTHOR: Erik Brorson
    """

    def __init__(self, lr=learning_rate_di_base, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=weight_decay, amsgrad=False,
                 multipliers=None, debug_verbose=False, **kwargs):
        super(Adam_lr_mult, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Learning rate multipliers
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None
            if multiplier:
                new_lr_t = lr_t * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
                    print(K.get_value(new_lr_t))
            else:
                new_lr_t = lr_t
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))
                    print(K.get_value(new_lr_t))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'multipliers': self.multipliers}
        base_config = super(Adam_lr_mult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def step_decay_schedule(initial_lr=learning_rate_di_base, decay_factor=learning_rate_decay_factor,
                        step_size=learning_rate_decay_epochs):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule, verbose=1)


adam_with_lr_multipliers = Adam_lr_mult(multipliers=learning_rate_multipliers)

mobile_model_multitask.compile(adam_with_lr_multipliers,
                               loss=['categorical_crossentropy', 'mean_squared_error'], metrics=['accuracy', 'mae'])

train_generator = data_generator_con_data_aug('lfw_train.csv', batch_size)
val_generator = data_generator_con_data_aug('lfw_val.csv', batch_size)
# test_generator= data_generator_prepr('chalearn_train.csv',1)


# callbacks
filepath = "mobilemultitaskDataAug.best.hdf5"
lr_sched = step_decay_schedule()
checkpoint = ModelCheckpoint(filepath, monitor='val_outA_mean_absolute_error', verbose=1, save_best_only=True)
callbacks_list = [lr_sched,
                  checkpoint]

mobile_model_multitask.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                                     callbacks=callbacks_list,
                                     validation_data=val_generator, validation_steps=validation_steps)

# scores = nas_model_multitask.evaluate_generator(test_generator,steps=test_size, verbose=1) #use_multiprocessing=True


# print("%s: %.2f%%" % (nas_model.metrics_names[1], scores[1] * 100))
