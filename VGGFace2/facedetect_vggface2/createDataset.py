import numpy as np
import cv2

from tqdm import tqdm

from VGGFace2.facedetect_vggface2.face_aligner import FaceAligner
from VGGFace2.facedetect_vggface2.face_detector import FaceDetector

fd = FaceDetector()
al = FaceAligner()


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


TIME_LOADING = 0
TIME_DETECTING = 0
TIME_CROPPING = 0
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
XML = open("/mnt/sdc1/acuissot/TrainXML.xml", "r")
content = XML.read()
idData, ethnicityData = parseXML(content)


def getCat(id):
    index = idData.index(id)
    return int(ethnicityData[index]) - 1


newPath = "/mnt/sdc1/acuissot/Faces_cropped_n_padded/testcd .."
categorizedPath = "/mnt/sdc1/acuissot/Faces_labeled/test"
for folder in tqdm(folders):
    cat = getCat(folder)
    folderPath = path + "/" + folder
    try:
        os.mkdir(newPath + "/" + folder)
    except OSError:
        pass

    if cat <= 3:
        images = os.listdir(folderPath)
        for image in images:
            imagePath = folderPath + "/" + image
            frame = cv2.imread(imagePath)
            faces = fd.detect(frame)
            images_processed += 1
            if len(faces) == 0:
                continue
            faces_found += 1
            f = findRelevantFace(faces, frame.shape[1], frame.shape[0])
            f['roi'] = enclosing_square(f['roi'])
            f['roi'] = add_margin(f['roi'], 0.2)
            img = cut(frame, f['roi'])
            cv2.imwrite(newPath + "/" + folder + "/" + image, img)
            cv2.imwrite(categorizedPath + "/" + str(cat) + "/" + folder + "_" + image, img)

print("Time loading: %f" % TIME_LOADING)
print("Time detecting: %f" % TIME_DETECTING)
print("Time cropping: %f" % TIME_CROPPING)
print("Faces found: %d/%d" % (faces_found, images_processed))
