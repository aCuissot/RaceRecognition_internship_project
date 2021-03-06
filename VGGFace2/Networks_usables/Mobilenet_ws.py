from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from PIL import Image

img_width, img_height = 224, 224
train_data_dir = '/mnt/sdc1/acuissot/Faces_labeled/train'
# validation_data_dir = "tf_files/codoon_photos"
nb_train_samples = 3141443
# nb_validation_samples = 466
batch_size = 64
epochs = 2
nb_classes = 4

model = applications.MobileNet(include_top=False, weights=None, input_shape=(img_width, img_height, 3))
# letting alpha = 1 (alpha < 1 reduce convolutional cost => faster but accuracy decrease too)
# Freeze the layers which you don't want to train. Here I am freezing the all layers.
for layer in model.layers[:]:
    layer.trainable = False

# Adding custom Layer
# We only add
x = model.output
x = Flatten()(x)
# Adding even more custom layers
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(1024, activation="relu")(x)
predictions = Dense(nb_classes, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

# compile the model
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     class_mode="categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("mobilenet_retrain.h5", monitor='jaaj', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='jaaj', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model
model_final.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    callbacks=[checkpoint, early])
# model_final.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     epochs=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples,
#     callbacks=[checkpoint, early])
