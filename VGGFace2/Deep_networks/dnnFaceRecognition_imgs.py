# Here we have an opencv DNN for face detection which detect a face and preprocess it thanks to dlib

from __future__ import division
import cv2
import time
import sys
import dlib

predictor_path = "C:\\Users\\Cuissot\\PycharmProjects\\untitled2\\VGGFace2\\shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# OpenCV DNN supports 2 networks.
# 1. FP16 version of the original caffe implementation ( 5.4 MB )
# 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

conf_threshold = 0.7


def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    frameOpencvDnn = cv2.cvtColor(frameOpencvDnn, cv2.COLOR_RGB2BGR)
    cv2.imshow("", frameOpencvDnn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frameOpencvDnn, bboxes


def faceDetectionOnImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, img)

    faces = dlib.full_object_detections()
    bbox = bboxes[0]
    faces.append(sp(img, dlib.rectangle(max(bbox[0], 0), max(bbox[1], 0), bbox[2], bbox[3])))
    image = dlib.get_face_chip(img, faces[0], size=224, padding=0.2)
    return image


face_file_path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train\\n000002\\0001_01.jpg"

imageTest = cv2.imread(face_file_path)
outOpencvDnn = faceDetectionOnImg(imageTest)
outOpencvDnn = cv2.cvtColor(outOpencvDnn, cv2.COLOR_RGB2BGR)
cv2.imshow("Face Detection and preprocessing", outOpencvDnn)
cv2.imshow("Original", imageTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
