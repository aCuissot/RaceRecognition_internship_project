import numpy as np
import cv2

from tqdm import tqdm

from VGGFace2.Networks_antonio_like.faceDetectionPart.face_detector import FaceDetector
from VGGFace2.Networks_antonio_like.faceDetectionPart.face_aligner import FaceAligner

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


def preprocessing_face_without_alignement(img_path):
    frame = cv2.imread(img_path)
    ##########
    faces = fd.detect(frame)
    if len(faces) == 0:
        print("no face detected")
        return frame, False
    # for f in faces:
    #    cv2.rectangle(frame, top_left(f), bottom_right(f), (0, 255, 0), 2)
    f = findRelevantFace(faces, frame.shape[1], frame.shape[0])
    # cv2.rectangle(frame, top_left(f), bottom_right(f), (255, 255, 0), 2)
    # cv2.imshow('img', frame)
    # cv2.waitKey(0)
    f['roi'] = enclosing_square(f['roi'])
    f['roi'] = add_margin(f['roi'], 0.2)
    img = cut(frame, f['roi'])
    return img, True
