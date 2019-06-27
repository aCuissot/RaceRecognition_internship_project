import cv2
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout, Dense

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# def convert_to_one_hot(Y, C):
#     Y = np.eye(C)[Y.reshape(-1)].T
#     return Y

# only a resize to match with the network
def preprocessing(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# create a data generator
# datagen = ImageDataGenerator(preprocessing_function=preprocessing)
datagen = ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('C:\\Users\\Cuissot\\PycharmProjects\\Data\\Network_test_dataset\\train',
                                       class_mode='binary', target_size=(224, 224), batch_size=64)
# load and iterate test dataset
# test_it = datagen.flow_from_directory('C:\\Users\\Cuissot\\PycharmProjects\\Data\\Network_test_dataset\\test',
#                                       class_mode='binary', target_size=(224, 224), batch_size=64)

batchX, batchy = train_it.next()
print(batchy)
# batchy = convert_to_one_hot(batchy, 6).T

print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
print('Batch labels shape=%s, min=%.3f, max=%.3f' % (batchy.shape, batchy.min(), batchy.max()))

num_classes = 4
model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model_final = Model(inputs=model.input, outputs=predictions)
from keras.optimizers import SGD, Adam

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model_final.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='mse')
print('jeej')
model_final.fit_generator(train_it, epochs=10, steps_per_epoch=108)
print('jaaj')
# evaluate model
# loss = model.evaluate_generator(test_it, steps=24)

img_path = 'C:\\Users\\Cuissot\\PycharmProjects\\untitled2\\VGGFace2\\Data\\aa_cropped.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', preds)
