import keras
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
from sklearn.metrics import confusion_matrix

img_width, img_height = 224, 224
test_data_dir = '/mnt/sdc1/acuissot/Faces_labeled/test'

nb_train_samples = 3141443
# nb_validation_samples = 466
batch_size = 64
epochs = 2
nb_classes = 4

model_final = keras.models.load_model("mobilenet_retrain.h5")
model_final.history
# Initiate the train and test generators with data Augumentation

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

# Train the model
eval = model_final.evaluate_generator(test_generator, batch_size=64)
predictions = model_final.predict_generator(test_datagen)
y_pred = (predictions > 0.5)
matrix = confusion_matrix(y_pred.labels.argmax(axis=1), test_generator.labels.argmax(axis=-1))
print(matrix)
save_matrix = open("confusion_matrix.txt", "w")
txt = "Evaluation\n" + str(eval)
txt += "\nConf Matrix:\n" + str(matrix)
save_matrix.write(txt)
