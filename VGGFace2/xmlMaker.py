import cv2 as cv
import numpy as np
import os
import sys

classesKeys = [ord("1"), ord("2"), ord("3"), ord("4")]
path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"


def getImage(index, imgs, f):
    imgPath = path + "\\" + f + "\\" + imgs[index]
    return cv.imread(imgPath)


def writeCategoryInXML(file, name, cat):
    file.write("<subject>\n")

    file.write("<curr_id>")
    file.write(str(name))
    file.write("</curr_id>\n")

    file.write("<ethnicity>")
    file.write(str(cat))
    file.write("</ethnicity>\n")

    file.write("</subject>\n")


def getMaxNumberPictureForSameId():
    folders = os.listdir(path)
    maxi = 0
    i = 0
    for f in folders:
        folderPath = path + "\\" + f
        images = os.listdir(folderPath)
        maxi = max(maxi, len(images))
        i += 1
        if i % 100 == 0:
            print(i)
    return maxi



# print(getMaxNumberPictureForSameId())  # 843 for training set, 761 for test => 843
# and min = 87 for training set and 98 for test => 87
folders = os.listdir(path)
XML = open("Data/labels/trainLabels.xml", "w+")
XML.write("<xml>\n")

print(len(folders))
startIndex = input("Index where you stopped last time")
startIndex = int(startIndex)
folders = folders[startIndex:]
curr_index = startIndex
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
            curr_index += 1
            break
        elif k == ord("s"):
            XML.write("</xml>")
            XML.close()
            print(curr_index)
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
