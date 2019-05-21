from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import PIL

# using ResNet50 to classify images:

model = ResNet50(weights='imagenet')

img_path = '../../../Data/VGGFacesV2/train/n000002/0021_01.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [('n01944390', 'snail', 0.99994576), ('n01945685', 'slug', 4.555212e-05), ('n01943899', 'conch', 2.6873558e-06)]
