# A first croping script using groundtruth as face box, no detection here
import os

import cv2 as cv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# import dlib

TrainSetPath = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"
TestSetPath = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test"


def getImage(imgId):
    imgPath = TrainSetPath + "\\" + imgId + ".jpg"

    return cv.imread(imgPath)


# Loading csv files (take few seconds)

csvIdentityMeta = pd.read_csv("Data/labels/new_identity_meta.csv")

# Train
csvBBGroundTruthTrainIds = pd.read_csv("Data/bb_landmark/loose_bb_train.csv").loc[:, "NAME_ID"]
csvBBGroundTruthTrain = pd.read_csv("Data/bb_landmark/loose_bb_train.csv").set_index("NAME_ID")
# csvLandmarkGroundTruthTrain = pd.read_csv("Data/bb_landmark/loose_landmark_train.csv")

"""
# Test
csvBBGroundTruthTestIds = pd.read_csv("Data/bb_landmark/loose_bb_test.csv").loc[:, "NAME_ID"]
csvBBGroundTruthTest = pd.read_csv("Data/bb_landmark/loose_bb_test.csv").set_index("NAME_ID")
csvLandmarkGroundTruthTest = pd.read_csv("Data/b_landmark/loose_landmark_test.csv")
"""

print(csvBBGroundTruthTrain.loc[csvBBGroundTruthTrainIds[0], :])
print(csvBBGroundTruthTrainIds[0])

print("Data loaded")

# Getting square ROI of each picture (face + padding of 20%)

folders = os.listdir(TrainSetPath)

folders = folders[:3]


# This function calculate the smallest size of a square which contain bb

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


# for imgId in csvBBGroundTruthTrainIds:
#     img = getImage(imgId)
#     imgOut = cropFace(img, imgId)


img = cv.imread(TrainSetPath + "\\n000002\\0018_01.jpg")
# cv.imshow("", img)
# cv.waitKey(0)
# cv.destroyAllWindows()
imgfacecrop = cropFace(img, "n000002/0018_01")
print(imgfacecrop.shape)
cv.imshow("", imgfacecrop)
cv.waitKey(0)
cv.destroyAllWindows()

"""
# Here we detect simply the face and eyes thanks to haarcascade date
face_cascade = cv.CascadeClassifier('Data/haar_detection/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('Data/haar_detection/haarcascade_eye.xml')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
"""
