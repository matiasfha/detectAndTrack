# encoding: utf-8
import cv2
import cv2.cv as cv
import numpy as np

class Detection:
    def __init__(self,cascade='haarcascade_frontalface_alt.xml'):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.cascade = cv2.CascadeClassifier(cascade)

    def people(self,image):
        found, w = self.hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.05)
        return found

    def faces(self,image):
        found = self.cascade.detectMultiScale(image,scaleFactor=1.3, minNeighbors=4, flags = cv2.CASCADE_SCALE_IMAGE,minSize=(30, 30))
        return found
