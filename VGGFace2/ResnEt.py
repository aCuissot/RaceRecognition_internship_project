from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()
# load and iterate training dataset
train_it = datagen.flow_from_directory("C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train", class_mode='binary')

model = ResNet50(weights='imagenet')
model.fit_generator(train_it, steps_per_epoch=16, validation_steps=8)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
