import cv2 as cv
import numpy as np
import os
import sys
import random

path = "/mnt/sdc1/acuissot/newLabelDataset/test/0"


def getImage(index, imgs, f):
    imgPath = path + "\\" + f + "\\" + imgs[index]
    return cv.imread(imgPath)


folders = os.listdir(path)

for i in range(int(len(folders) / 1000)):

    rndimg = random.choice(folders)
    index = 0

    img = getImage(index, rndimg, folders)
    cv.imshow(str(folders), img)
    k = cv.waitKey(0)

    if k == ord("0"):
        print("jeej")
        break
    elif k == ord("q"):
        sys.exit()

cv.destroyAllWindows()
