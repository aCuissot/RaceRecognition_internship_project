import os

import keras
from keras.utils import Sequence, multi_gpu_model
from keras.preprocessing import image
import numpy as np
import cv2 as cv
import pandas as pd
from keras_applications.vgg16 import VGG16
from keras_applications.xception import Xception

csvBBGroundTruthTrainIds = pd.read_csv("../Data/bb_landmark/loose_bb_train.csv").loc[:, "NAME_ID"]
csvBBGroundTruthTrain = pd.read_csv("../Data/bb_landmark/loose_bb_train.csv").set_index("NAME_ID")


def getPrimarySquareSize(shape, bb):
    maxBB = max(bb[2], bb[3])
    if maxBB > shape[0] or maxBB > shape[1]:
        return min(shape[0], shape[1])
    else:
        return maxBB


# Here we crop the face in a square image with a padding of 20% if it's possible

def cropFace(img, imgId):
    shape = img.shape
    # print(shape)
    bb = csvBBGroundTruthTrain.loc[imgId, :]
    # print(bb)
    primarySquareSize = getPrimarySquareSize(shape, bb)
    # print(primarySquareSize)
    topleft = [int(bb[0] - ((primarySquareSize - bb[2]) / 2) - (0.2 * primarySquareSize)),
               int(bb[1] - ((primarySquareSize - bb[3]) / 2) - (0.2 * primarySquareSize))]
    botright = [int(bb[0] + primarySquareSize - ((primarySquareSize - bb[2]) / 2) + (0.2 * primarySquareSize)),
                int(bb[1] + primarySquareSize - ((primarySquareSize - bb[3]) / 2) + (0.2 * primarySquareSize))]
    # print(topleft)
    # print(botright)
    topleft[0] = max(topleft[0], 0)
    topleft[1] = max(topleft[1], 0)
    botright[0] = min(botright[0], shape[0])
    botright[1] = min(botright[1], shape[1])
    # print(topleft)
    # print(botright)
    # out = img.copy()
    # cv.rectangle(out, (topleft[0], topleft[1]), (botright[0], botright[1]), (255, 0, 0), 3)
    # cv.rectangle(out, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 0), 3)
    # out = out[topleft[1]:botright[1], topleft[0]:botright[0]]

    # return out
    return img[topleft[1]:botright[1], topleft[0]:botright[0]]


def preprocessing(model, img_name, img_path):
    targetSize = (224, 224)  # works for mobilenet, resnet and vgg16
    if model == 'nasnet':
        targetSize = (331, 331)
    img = cv.imread(img_path)
    img = cropFace(img, img_name)
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
    return x


def loadImages(model, trainBool=True):
    path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"
    if not trainBool:
        path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test"

    X_train = []
    folders = os.listdir(path)
    folders = folders[:200]
    index = 0
    for f in folders:
        folderPath = path + "\\" + f
        images = os.listdir(folderPath)
        index += 1
        if index % 10 == 0:
            print(index / 2)
        for imageName in images:
            image_path = folderPath + "\\" + imageName
            newImage = preprocessing(model, f + "/" + imageName.split(".")[0], image_path)
            X_train.append(newImage)  # bad idea


from skimage.io import imread
from skimage.transform import resize


class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
            for file_name in batch_x]), np.array(batch_y)


batch_size = 64
num_epochs = 20
num_training_samples = 1868
num_validation_samples = 76


def getIDsFromFile(file, FullPath):
    idsList = []
    for line in file:
        idsList.append(FullPath + "\\" + line)
    return idsList


trainSetIds = open('../Data/labels/homogeneousTrainSetIds.txt', "r")
testSetIds = open('../Data/labels/homogeneousTestSetIds.txt', "r")
training_filenames = getIDsFromFile(trainSetIds, "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train")

validation_filenames = getIDsFromFile(testSetIds, "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train")


def getEthListFromFile(file):
    ethList = []
    for line in file:
        ethList.append(line)


trainLabels = open('../Data/labels/homogeneousTrainLabels.txt', "r")
testLabels = open('../Data/labels/homogeneousTestLabels.txt', "r")

GT_training = getEthListFromFile(trainLabels)
GT_validation = getEthListFromFile(testLabels)

my_training_batch_generator = My_Generator(training_filenames, GT_training, batch_size)
my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size)

model = VGG16(weights=None)

model.fit_generator(generator=my_training_batch_generator,
                    steps_per_epoch=(num_training_samples // batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    validation_data=my_validation_batch_generator,
                    validation_steps=(num_validation_samples // batch_size),
                    use_multiprocessing=True,
                    workers=16,
                    max_queue_size=32)

loadImages('')
