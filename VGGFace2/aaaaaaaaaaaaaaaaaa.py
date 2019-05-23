import cv2 as cv


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


# cvMethod("Data/aa.jpg")
handMethod("Data/aa.jpg")
