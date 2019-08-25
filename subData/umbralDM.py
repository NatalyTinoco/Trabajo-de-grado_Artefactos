# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:24:21 2019

@author: Nataly
"""
import cv2 
import glob 
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS')

#from readimg import read_img
import numpy as np
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt
from rOI import ROI


for imgfile in glob.glob("*.jpg"):
    imgfile='00054.jpg'
    img=cv2.imread(imgfile)   
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    imaROI=ROI(img)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
      