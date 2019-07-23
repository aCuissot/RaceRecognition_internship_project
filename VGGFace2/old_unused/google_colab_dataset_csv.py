# Creating a csv file adapted to collab versio of the dataset
import os
import cv2 as cv
from VGGFace2.utils.xmlParserFnct import *

path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"


def getImagePath(index, imgs, f):
    imgPath = path + "\\" + f + "\\" + imgs[index]
    return imgPath


folders = os.listdir(path)
folders = folders[:94]
XML = open("Data/labels/TrainXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)

txt = ""
for f in folders:

    folderPath = path + "\\" + f
    images = os.listdir(folderPath)
    for img in images:
        i = idData.index("<id>" + f + "</id>")
        # if int(ethnicityData[i])
        txt += f + '/' + img.split('.')[0] + ", " + ethnicityData[i] + "\n"

destFile = open("Data/labels/colab.csv", "w")
destFile.write(txt)
destFile.close()

cv.destroyAllWindows()
XML.close()
