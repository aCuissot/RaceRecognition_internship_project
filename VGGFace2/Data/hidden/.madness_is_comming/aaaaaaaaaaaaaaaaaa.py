import cv2 as cv
import numpy as np


def cvMethod(fileName, outputName="aaaaaa.txt"):
    img = cv.imread(fileName)
    img = cv.resize(img, (80, 80))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    tot = ""
    line = ""
    for i in ret:
        for j in i:
            if j == 0:
                line += "  "
            else:
                line += "aa"
        tot += line + "\n"
        line = ""
    file = open(outputName, "w")
    file.write(tot)
    file.close()


def handMethod(fileName, outputName="aaaaaa.txt"):
    img = cv.imread(fileName)
    img = cv.resize(img, (80, 80))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    tot = ""
    line = ""
    for i in img:
        for j in i:
            if j < 255 / 4:
                line += "  "
            elif j < 255 / 2:
                line += "aa"
            elif j < 3 * 255 / 4:
                line += "AA"
            else:
                line += "@@"
        tot += line + "\n"
        line = ""
    file = open(outputName, "w")
    file.write(tot)
    file.close()


def grayLvlMethod(fileName, outputName="aaaaaa.txt"):
    ESCAPE = "\x1b"
    BLACK = "[30m"
    BLUE = "[34m"
    RED = "[31m"
    ddepth = cv.CV_16S
    kernel_size = 3
    blank_image = np.zeros((80, 80, 1), np.uint8)
    for i in blank_image:
        for j in i:
            j[0] = 127
    img = cv.imread(fileName)
    img = cv.resize(img, (80, 80))
    src = cv.GaussianBlur(img, (3, 3), 0)
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(src, ddepth, kernel_size)
    dst = cv.convertScaleAbs(dst)
    tot = 0
    for i in range(80):
        for j in range(80):
            tot += dst[i][j]
    average = int(tot / 6400)
    for i in range(80):
        for j in range(80):
            if dst[i][j] < average:
                dst[i][j] = 0
            else:
                dst[i][j] = 1
    txt = ""
    line = ""
    for i in range(80):
        for j in range(80):
            if dst[i][j] == 0:
                line += ESCAPE + BLUE + "aa"
            elif img[i][j] > 127:
                line += ESCAPE + RED + "aa"
            else:
                line += "  "
        txt += line + "\n"
        line = ""
    print(txt)


# cvMethod("Data/aa.jpg")
# handMethod("Data/aa.jpg")
ESCAPE = "\x1b"
BLACK = "[30m"
BLUE = "[34m"
RED = "[31m"

grayLvlMethod("Data/aa.jpg")
