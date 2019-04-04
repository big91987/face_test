import os
import sys

from dataset.faceRecognition.faceCorrection import *
from dataset.faceRecognition import faceDetectionLib as detect_face
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


load_prefix = './dataset/CASIA-WebFace/'
save_prefix = './dataset/CASIA-WebFace-cut/'

if __name__ == '__main__':
    with open('./casia_landmark.txt', 'r') as f:
        for line in f:
            print('line = {}'.format(line))
            tmp = line.split()
            img_name = tmp[0]
            id = int(tmp[1])
            landmarks = tmp[2:]

            print('image_name = {}, id = {}, landmarks = {}'.format(img_name, id, landmarks))

            img_path = load_prefix + img_name
            if os.path.exists(img_path):
                pass