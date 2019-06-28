import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


# cap = cv2.VideoCapture(0)

def top_left(f):
    return (f['roi'][0], f['roi'][1])


def bottom_right(f):
    return (f['roi'][0] + f['roi'][2], f['roi'][1] + f['roi'][3])


def enclosing_square(rect):
    def _to_wh(s, l, ss, ll, width_is_long):
        if width_is_long:
            return l, s, ll, ss
        else:
            return s, l, ss, ll

    def _to_long_short(rect):
        x, y, w, h = rect
        if w > h:
            l, s, ll, ss = x, y, w, h
            width_is_long = True
        else:
            s, l, ss, ll = x, y, w, h
            width_is_long = False
        return s, l, ss, ll, width_is_long

    s, l, ss, ll, width_is_long = _to_long_short(rect)

    hdiff = (ll - ss) // 2
    s -= hdiff
    ss = ll

    return _to_wh(s, l, ss, ll, width_is_long)


def add_margin(roi, qty):
    return (
        roi[0] - qty,
        roi[1] - qty,
        roi[2] + 2 * qty,
        roi[3] + 2 * qty)


def cut(frame, roi):
    pA = (int(roi[0]), int(roi[1]))
    pB = (int(roi[0] + roi[2] - 1), int(roi[1] + roi[3] - 1))  # pB will be an internal point
    W, H = frame.shape[1], frame.shape[0]
    A0 = pA[0] if pA[0] >= 0 else 0
    A1 = pA[1] if pA[1] >= 0 else 0
    data = frame[A1:pB[1], A0:pB[0]]
    if pB[0] < W and pB[1] < H and pA[0] >= 0 and pA[1] >= 0:
        return data
    w, h = int(roi[2]), int(roi[3])
    img = np.zeros((h, w, 3), dtype=np.uint8)
    offX = int(-roi[0]) if roi[0] < 0 else 0
    offY = int(-roi[1]) if roi[1] < 0 else 0
    np.copyto(img[offY:offY + data.shape[0], offX:offX + data.shape[1]], data)
    return img


def findRelevantFace(objs, W, H):
    mindistcenter = None
    minobj = None
    for o in objs:
        cx = o['roi'][0] + (o['roi'][2] / 2)
        cy = o['roi'][1] + (o['roi'][3] / 2)
        distcenter = (cx - (W / 2)) ** 2 + (cy - (H / 2)) ** 2
        if mindistcenter is None or distcenter < mindistcenter:
            mindistcenter = distcenter
            minobj = o
    return minobj


faces_found = 0
images_processed = 0

import sys
import os
from time import time

"""
fin_name = sys.argv[1]
img_path = os.path.join( os.path.dirname(fin_name), os.path.basename(fin_name)[:-4] )
fout_name = os.path.basename(fin_name)[:-4]+'.detected.txt'
fin = open(fin_name, "r")
fout = open(fout_name, "w")
for line in tqdm(fin):
"""


def getId(list):
    sublist = []
    n = len(list)
    for i in range(0, n - 1, 2):
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
    xmlStr = xmlStr.replace("</id>", "")
    xmlStr = xmlStr.replace("<id>", "")

    xmlStr = xmlStr.replace("<curr_id>", "")
    xmlStr = xmlStr.replace("<ethnicity>", "")
    xmlStr = xmlStr.replace("</curr_id>", "")
    xmlStr = xmlStr.replace("</ethnicity>", "")

    list = xmlStr.split("\n")
    id = getId(list)
    ethnicity = getEthnicity(list)
    return id, ethnicity



path = "/mnt/sdc1/acuissot/original_dataset/test"
folders = os.listdir(path)
XML = open("/mnt/sdc1/acuissot/TestXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)


def getCat(id):
    index = idData.index(id)
    return int(ethnicityData[index]) - 1


newPath = "/mnt/sdc1/acuissot/Faces_cropped_n_padded/test"
categorizedPath = "/mnt/sdc1/acuissot/Faces_labeled/test"


def getCsvContent(csvPath):
    csv = pd.read_csv(csvPath)
    csv.drop(csv.columns[[0, 1]], axis=1, inplace=True)
    print(csv)
    return csv


csvpath = "/mnt/sdc1/acuissot/test.detected1.csv"
csvContent = getCsvContent(csvpath)


def getFaces(row):
    x, y, w, h = row[2], row[3], row[4], row[5]
    return x, y, w, h



for index, row in csvContent.iterrows():
    img_id = row[0]
    folder = img_id.split('/')[0]
    cat = getCat(folder)
    folderPath = path + "/" + folder
    try:
        os.mkdir(newPath + "/" + folder)
    except OSError:
        pass

    if cat <= 3:

        imagePath = path + "/" + img_id
        frame = cv2.imread(imagePath)
        # faces = fd.detect(frame)
        face = getFaces(row)
        face = enclosing_square(face)
        face = add_margin(face, 0.2)
        img = cut(frame, face)
        # cv2.imshow("original", frame)
        # cv2.imshow("done", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(newPath + "/" + folder + "/" + image, img)
        cv2.imwrite(categorizedPath + "/" + str(cat) + "/" + folder + "_" + image, img)

print("Faces found: %d/%d" % (faces_found, images_processed))
