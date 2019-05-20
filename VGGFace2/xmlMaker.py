import cv2 as cv
import numpy as np
import os
import sys

classesKeys = [ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")]
path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"


def getImage(index, imgs, f):
    imgPath = path + "\\" + f + "\\" + imgs[index]
    return cv.imread(imgPath)


def writeCategoryInXML(file, name, cat):
    file.write("<subject>\n")

    file.write("<id>")
    file.write(str(name))
    file.write("</id>\n")

    file.write("<ethnicity>")
    file.write(str(cat))
    file.write("</ethnicity>\n")

    file.write("</subject>\n")


folders = os.listdir(path)
XML = open("Data/labels/trainLabels1.xml", "w+")
XML.write("<xml>\n")

print(len(folders))
folders = folders[:200]
for f in folders:

    folderPath = path + "\\" + f
    images = os.listdir(folderPath)

    index = 0
    while True:

        img = getImage(index, images, f)
        cv.imshow(str(f), img)
        k = cv.waitKey(0)

        if k in classesKeys:
            cat = classesKeys.index(k) + 1
            writeCategoryInXML(XML, f, cat)
            break
        elif k == ord("q"):
            XML.write("</xml>")
            XML.close()
            sys.exit()
        else:
            if index < len(images):
                index += 1
            else:
                index = 0
    cv.destroyAllWindows()
cv.destroyAllWindows()
XML.write("</xml>")
XML.close()
