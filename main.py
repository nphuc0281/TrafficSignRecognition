import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os

from traffic_sign_recognition import detect

# Initialization data
class_number = 52
cur_path = os.getcwd()


if __name__ == '__main__':
    # Load class names.
    classesFile = r"data\models\traffic_sign.names"
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the weight files to the model and load the network using them.
    modelWeights = r"data\models\traffic_sign.onnx"
    net = cv2.dnn.readNet(modelWeights)

    frame = cv2.imread(r'data\realtime_images\IMG (2).jpg')
    img, label = detect.detect(frame, classes, net)

    cv2.imshow('Output', img)
    cv2.waitKey(0)
