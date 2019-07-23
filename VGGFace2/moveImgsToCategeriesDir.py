import os
import random
import shutil
from VGGFace2.utils.xmlParserFnct import *

path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\test"
newPath = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFaceV2\\test"
folders = os.listdir(path)
random.shuffle(folders)

# print(folders)

folders = os.listdir(path)
XML = open("Data/labels/TestXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)


def getCat(id):
    index = idData.index(id)
    return int(ethnicityData[index]) - 1


index = 0

total = 0
for f in folders:
    cat = getCat(f)
    folderPath = path + "\\" + f

    if cat == 0:
        # images = os.listdir(folderPath)
        total += 1
    #     for image in images:
    #         imagePath = folderPath + "\\" + image
    #         shutil.copyfile(imagePath, newPath + "\\" + str(cat) + "\\" + f + "_" + image)
    #     index += 1
    #     if index == 8000:
    #         print("end of validation")
    #         path.replace("train", "validation")
    #     if index % 80 == 0:
    #             print("%d/100" % (index // 80))

print(total)
