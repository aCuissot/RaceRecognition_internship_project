import numpy as np
import cv2

from tqdm import tqdm

import os


def getId(list):
    sublist = []
    n = len(list)
    for i in range(0, n - 1, 2):
        sublist.append(list[i])
    return sublist


def getEthnicity(list):
    sublist = []
    n = len(list)
    for i in range(1, n, 2):
        sublist.append(list[i])
    return sublist


def parseXML(xmlStr):
    xmlStr = xmlStr.replace("<xml>\n", "")
    xmlStr = xmlStr.replace("</xml>", "")
    xmlStr = xmlStr.replace("<subject>\n", "")
    xmlStr = xmlStr.replace("</subject>\n", "")
    xmlStr = xmlStr.replace("</id>", "")
    xmlStr = xmlStr.replace("<id>", "")

    xmlStr = xmlStr.replace("<curr_id>", "")
    xmlStr = xmlStr.replace("<ethnicity>", "")
    xmlStr = xmlStr.replace("</curr_id>", "")
    xmlStr = xmlStr.replace("</ethnicity>", "")

    list = xmlStr.split("\n")
    id = getId(list)
    ethnicity = getEthnicity(list)
    return id, ethnicity


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
