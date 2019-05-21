from keras.applications.nasnet import preprocess_input, decode_predictions, NASNetLarge
from keras.preprocessing import image
import numpy as np

model = NASNetLarge(weights='imagenet')

img_path = '../../../Data/VGGFacesV2/train/n000002/0021_01.jpg'
img = image.load_img(img_path, target_size=(331, 331))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
