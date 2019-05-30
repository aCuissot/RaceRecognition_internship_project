import os
from multiprocessing import freeze_support

import cv2 as cv
import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.preprocessing import image
from keras.utils import Sequence
from keras_applications.resnet50 import preprocess_input
from skimage.io import imread
from skimage.transform import resize

csvBBGroundTruthTrainIds = pd.read_csv("../Data/bb_landmark/loose_bb_train.csv").loc[:, "NAME_ID"]
csvBBGroundTruthTrain = pd.read_csv("../Data/bb_landmark/loose_bb_train.csv").set_index("NAME_ID")
Path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\"


def getFileElementsList(file, prefix):
    filecontent = open(file, "r")
    contentStr = filecontent.read()
    filecontent.close()
    list = contentStr.split('\n')

    for i_loc in range(len(list)):
        list[i_loc] = prefix + list[i_loc]
    return list


# idTrainList = getFileElementsList('../Data/labels/homogeneousTrainSetIds.txt', 'train\\')
# idTestList = getFileElementsList('../Data/labels/homogeneousTestSetIds.txt', 'test\\')
idTrainList = getFileElementsList('../Data/labels/homogeneousTrainImgs.txt', 'train\\')
idTestList = getFileElementsList('../Data/labels/homogeneousTestImgs.txt', 'test\\')
labelTrainList = getFileElementsList('../Data/labels/homogeneousTrainLabels.txt', "")
labelTestList = getFileElementsList('../Data/labels/homogeneousTestLabels.txt', "")
partition = {'train': idTrainList,
             'validation': idTestList
             }
labels = {}

curr_id = ""
index = -1
for i in idTrainList:
    if curr_id != i.split("\\")[1][-4:]:
        index += 1
        curr_id = i.split("\\")[1][-4:]

    labels[i] = labelTrainList[index]
index = -1
for i in idTestList:
    if curr_id != i.split("\\")[1][-4:]:
        index += 1
        curr_id = i.split("\\")[1][-4:]
    labels[i] = labelTestList[index]

# Only for debugging

# aaaaaa = open("tmp.txt", "w")
# aaaaaa.write(str(labels))
# aaaaaa.close()
# bbbbbb = open("tmp2.txt", "w")
# for i in labels:
#     bbbbbb.write(i + "\n")
# bbbbbb.close()

# end of debug session


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


def preprocessing(model, img_name, img_path):
    targetSize = (224, 224)  # size for mobilenet, resnet and vgg16
    if model == 'nasnet':
        targetSize = (331, 331)
    img = cv.imread(img_path)
    img_id = img_path.split("\\")[-2] + "/" + img_name.split("\\")[-1]
    img = cropFace(img, img_id)
    img = cv.resize(img, targetSize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if model == 'mobilenet':
        x = keras.applications.mobilenet.preprocess_input(x)
    elif model == 'resnet':
        x = keras.applications.resnet50.preprocess_input(x)
    elif model == 'vgg16':
        x = keras.applications.vgg16.preprocess_input(x)
    elif model == 'nasnet':
        x = keras.applications.nasnet.preprocess_input(x)
    else:
        raise Exception(
            'model parameter have to be \'mobilenet\', \'resnet\', \'vgg16\' or \'nasnet\', here it is : {}'.format(
                model))
    return x
    # x shape = (1, 224, 224, 3)....


"""
def loadImages(model, trainBool=True):
    path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"
    if not trainBool:
        path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test"

    X_train = []
    folders = os.listdir(path)
    folders = folders[:200]
    for f in folders:
        folderPath = path + "\\" + f
        images = os.listdir(folderPath)
        for imageName in images:
            image_path = folderPath + "\\" + imageName
            newImage = preprocessing(model, f + "/" + imageName.split(".")[0], image_path)
            X_train.append(newImage)  # bad idea => generator
"""


# don't forgot changing dim when using NASNet


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, labels, model, batch_size=5, dim=(224, 224), n_channels=3,
                 n_classes=10, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.model = model
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        # np.empty create an array of the dimensions given, but data is not initialize (unlike np.zeros) => faster
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        advencement = 0
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            imgPath = Path + ID

            # Store sample
            X[i,] = preprocessing(self.model, imgPath.split(".")[0], imgPath)
            advencement += 1
            if advencement % 100 == 0:
                print(str(advencement / 10) + "%")
            # Store class
            y[i] = self.labels[ID]
        print("DataGeneration")
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def main():
    # training resnet and testing it
    params = {'dim': (1, 224, 224),
              'model': 'resnet',
              'batch_size': 1000,
              'n_classes': 4,
              'n_channels': 3,
              'shuffle': True}

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    from keras.applications.resnet50 import ResNet50

    model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        workers=1)
    # workers allow us to create different threads to treat batches in parallel
    # model.summary()

    # preparing an image for testing model
    img = image.load_img("../Data/test.jpg", target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(model.predict(x))


if __name__ == '__main__':
    freeze_support()
    main()
