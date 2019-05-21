from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.preprocessing import image
import numpy as np
import keras

# Using VGG16 to extract features:

model = VGG16(weights='imagenet')
img_path = '../../../Data/VGGFacesV2/train/n000002/0021_01.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

label = decode_predictions(features)
print(label)
