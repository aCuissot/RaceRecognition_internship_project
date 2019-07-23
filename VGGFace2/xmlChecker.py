import cv2 as cv
import numpy as np
import os
import sys
from VGGFace2.utils.xmlParserFnct import *

classesKeys = [ord("0"), ord("1")]
path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"


def getImage(index, imgs, f):
    imgPath = path + "\\" + f + "\\" + imgs[index]
    return cv.imread(imgPath)


category = 1

folders = os.listdir(path)
XML = open("Data/labels/TrainXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)


def showProportion(ethnicityData):
    Af, As, La, In, Ar, Ot = 0, 0, 0, 0, 0, 0
    for i in ethnicityData:
        tmp = int(i)
        if tmp == 1:
            Af += 1
        elif tmp == 2:
            As += 1
        elif tmp == 3:
            La += 1
        elif tmp == 4:
            In += 1
        elif tmp == 6:
            Ot += 1
        else:
            print("erreur" + str(tmp))
    tot = Af + Ar + As + La + In + Ot
    print(Af / tot)
    print(As / tot)
    print(La / tot)
    print(In / tot)
    print(Ar / tot)
    print(Ot / tot)
    print(Af)
    print(As)
    print(La)
    print(In)
    print(Ar)
    print(Ot)

    print(tot)


# showProportion(ethnicityData)
# folders = folders[6000:7000]
print(idData)

for f in folders:

    folderPath = path + "\\" + f
    images = os.listdir(folderPath)

    i = idData.index(f)
    # print(i)
    # print(ethnicityData[i])

    if int(ethnicityData[i]) == category:

        index = 0
        while True:

            img = getImage(index, images, f)
            cv.imshow(str(f), img)
            k = cv.waitKey(0)

            if k in classesKeys:
                cat = classesKeys.index(k)
                if cat == 1:
                    print(f)
                    newCat = cv.waitKey(0)
                    print(newCat - 48)

                break
            elif k == ord("q"):
                XML.close()
                sys.exit()
            else:
                if index < len(images):
                    index += 1
                else:
                    index = 0
        cv.destroyAllWindows()

cv.destroyAllWindows()
XML.close()
