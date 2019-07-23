import numpy as np
import cv2
from VGGFace2.utils.xmlParserFnct import *
from tqdm import tqdm
import os

path = "/mnt/sdc1/acuissot/Faces_labeled/test"

folders = os.listdir(path)
XML = open("/mnt/sdc1/acuissot/faceDetectScripts/NewTestXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)


def getCat(identity):
    index = idData.index(identity)
    return int(ethnicityData[index]) - 1


from shutil import copyfile

newPath = "/mnt/sdc1/acuissot/newLabelDataset"
for folder in tqdm(folders):
    folderPath = path + "/" + folder
    images = os.listdir(folderPath)
    for image in images:
        # identity = image.split("_")[0][1:]
        # cat = getCat(identity)
        # imagePath = folderPath + "/" + image
        # img = cv2.imread(imagePath)
        # cv2.imwrite(newPath + "/" + str(cat) + "/" + image, img)

        identity = image.split("_")[0][1:]
        cat = getCat(identity)
        copyfile(folderPath + "/" + image, newPath + "/" + str(cat) + "/" + image)
