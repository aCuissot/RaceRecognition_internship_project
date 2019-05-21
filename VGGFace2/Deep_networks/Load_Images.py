import os
from keras.preprocessing import image
import numpy as np
import cv2 as cv
import pandas as pd

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
    return cv.resize(img, targetSize)


def loadImages(model, trainBool=True):
    path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"
    if not trainBool:
        path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test"

    X_train = []
    folders = os.listdir(path)

    for f in folders:
        folderPath = path + "\\" + f
        images = os.listdir(folderPath)

        for imageName in images:
            image_path = folderPath + "\\" + imageName
            newImage = preprocessing(model, f + "/" + imageName.split(".")[0], image_path)
            # cv.imshow("", newImage)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            X_train.append(newImage)


loadImages('')
