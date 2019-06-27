from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
datagen = ImageDataGenerator()
# load and iterate training dataset
train_it = datagen.flow_from_directory("C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train", class_mode='binary')
print(train_it)
val_it = datagen.flow_from_directory("C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test", class_mode='binary')
test_it = datagen.flow_from_directory("C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test", class_mode='binary')
model = VGG16(weights='imagenet', include_top=False, classes=4)
model.compile()
model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)
loss = model.evaluate_generator(test_it, steps=24)
# yhat = model.predict_generator(predict_it, steps=24)
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
print(batchX)
# example of progressively loading images from file
# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset


# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))