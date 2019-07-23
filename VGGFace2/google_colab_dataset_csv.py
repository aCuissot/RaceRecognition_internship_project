# Creating a csv file adapted to collab versio of the dataset
import os
import cv2 as cv

path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"


def getImagePath(index, imgs, f):
    imgPath = path + "\\" + f + "\\" + imgs[index]
    return imgPath


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

    xmlStr = xmlStr.replace("<curr_id>", "")
    xmlStr = xmlStr.replace("<ethnicity>", "")
    xmlStr = xmlStr.replace("</curr_id>", "")
    xmlStr = xmlStr.replace("</ethnicity>", "")

    list = xmlStr.split("\n")
    id = getId(list)
    ethnicity = getEthnicity(list)
    return id, ethnicity


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
