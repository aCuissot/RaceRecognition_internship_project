import cv2 as cv
import numpy as np
import os
import sys
import random

classesKeys = [ord("0"), ord("1")]
path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"


def getImage(index, imgs, f):
    imgPath = path + "\\" + f + "\\" + imgs[index]
    return cv.imread(imgPath)


def getId(list):
    sublist = []
    n = len(list)
    for i in range(0, n, 2):
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

    xmlStr = xmlStr.replace("<id>", "")
    xmlStr = xmlStr.replace("<ethnicity>", "")
    xmlStr = xmlStr.replace("</id>", "")
    xmlStr = xmlStr.replace("</ethnicity>", "")

    list = xmlStr.split("\n")
    id = getId(list)
    ethnicity = getEthnicity(list)
    return id, ethnicity


folders = os.listdir(path)
XML = open("Data/labels/TrainXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)

Categories = [[], [], [], []]
for f in folders:
    i = idData.index(f)
    if int(ethnicityData[i]) == 1:
        Categories[0].append(f)
    elif int(ethnicityData[i]) == 2:
        Categories[1].append(f)
    elif int(ethnicityData[i]) == 3:
        Categories[2].append(f)
    elif int(ethnicityData[i]) == 4:
        Categories[3].append(f)

lens = [len(Categories[0]), len(Categories[1]), len(Categories[2]), len(Categories[3])]
smallest_category_number = min(lens)
finalset = []

for category in range(4):
    for i in range(smallest_category_number):
        finalset.append(Categories[category][random.randint(0, lens[category]-1)])

print(finalset)
random.shuffle(finalset)
print(finalset)
listSave = open("Data/labels/homogeneousSetIds.txt", "w")
txt = ""
for id in finalset:
    txt += id + "\n"
listSave.write(txt)
listSave.close()
XML.close()
