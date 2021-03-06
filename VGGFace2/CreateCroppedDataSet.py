# This script detect face on images, crop face and add a padding

import os
import random
import shutil
import cv2
import dlib
from VGGFace2.utils.xmlParserFnct import *

path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test"
# newPath = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFaceV2\\test"
folders = os.listdir(path)
random.shuffle(folders)

# print(folders)

modelFile = "Deep_networks/models/opencv_face_detector_uint8.pb"
configFile = "Deep_networks/models/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

conf_threshold = 0.7


def faceDetectionOnImg(img):
    """
    :param img: The given image
    :return: the cropped face with padding and a boolean to know if face is correctly xtracted
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = detectFaceOpenCVDnn(img)
    # outOpencvDnn = cv2.cvtColor(outOpencvDnn, cv2.COLOR_RGB2BGR)
    # print(bboxes)
    if bboxes:
        # if there s multiple faces detected, just keeping the first one
        img = cropFaceWithPadding(img, bboxes[0])

        return img, True
    else:

        return img, False


def getPrimarySquareSize(shape, bb):
    """
    :param shape: the shape of the image
    :param bb: the detected face rectangle
    :return: the size of the smallest square in the image containing the face
    """
    maxBB = max(bb[2], bb[3])
    if maxBB > shape[0] or maxBB > shape[1]:
        return min(shape[0], shape[1])
    else:
        return maxBB


def cropFaceWithPadding(img, faceDetected):
    """
    :param img: the image to crop
    :param faceDetected: the face detected box
    :return: a croped image containg the face
    """
    # print(faceDetected)
    shape = img.shape
    # bb = csvBBGroundTruthTrain.loc[imgId, :]
    primarySquareSize = getPrimarySquareSize(shape, faceDetected)
    width = faceDetected[2] - faceDetected[0]
    height = faceDetected[3] - faceDetected[1]
    topleft = [int(faceDetected[0] - (0.2 * width)), int(faceDetected[1] - (0.2 * height))]
    botright = [int(faceDetected[0] + width + (0.2 * width)), int(faceDetected[1] + height + (0.2 * height))]

    topleft[0] = max(topleft[0], 0)
    topleft[1] = max(topleft[1], 0)
    botright[0] = min(botright[0], shape[0])
    botright[1] = min(botright[1], shape[1])

    return img[topleft[1]:botright[1], topleft[0]:botright[0]]


def detectFaceOpenCVDnn(frame):
    """
    :param frame: the image  with a face
    :return: the boxes of the face
    """
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            # cv2.imshow("detector", frameOpencvDnn)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    return bboxes


def preprocess(img):
    """
    :param img: the image
    :return: the cropped face if it's possible, else the input image, a boolean to check if all worked
    """
    if img is not None:
        return faceDetectionOnImg(img)
    else:
        return img, False





folders = os.listdir(path)
XML = open("Data/labels/TestXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)


def getCat(id):
    """
    :param id: the id
    :return: the ethnicity as it will be used in the labeled dataset
    """
    index = idData.index(id)
    return int(ethnicityData[index]) - 1


index = 0

total = 0

# Here we read all the images of the set, preprocess them and create 2 new dataset:
# One with the same structure than the original with the cropped faces
# an other with a new structure: images are classified by label

newPath = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\Faces_cropped_and_padded\\test"
categorizedPath = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\Faces_labeled\\test"
for f in folders:
    cat = getCat(f)
    folderPath = path + "\\" + f
    try:
        os.mkdir(newPath + "\\" + f)
    except OSError:
        pass

    if cat <= 3:
        images = os.listdir(folderPath)
        for image in images:
            imagePath = folderPath + "\\" + image
            img = cv2.imread(imagePath)
            # cv2.imshow('Original', img)
            cropped, verifier = preprocess(img)
            if verifier:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cv2.imwrite(newPath + "\\" + f + "\\" + image, cropped)
                cv2.imwrite(categorizedPath + "\\" + str(cat) + "\\" + f + "_" + image, cropped)
            # open img, preprocess it n save it in two places
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        index += 1
        # if index == 8000:
        #     print("end of validation")
        if index % 5 == 0:
            print("%d/100" % (index // 80))

print(total)
