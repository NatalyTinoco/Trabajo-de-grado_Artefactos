# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:17:01 2019

@author: Usuario
"""

import sys
path = "C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE"
sys.path.insert(1, path)

import cv2
import glob

im = cv2.imread('00031-14.jpg',0)

areas = []
perimeter = []
compacidad = []

for image in glob.glob("*.jpg"):
    im = cv2.imread(image,0)
    imNorm = cv2.normalize(im,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
    
    contours,hierarchy = cv2.findContours(imNorm, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        for c in range(len(contours)):
            cnt = contours[c]
            areas.append(cv2.contourArea(cnt))
            perimeter.append(cv2.arcLength(cnt,True))
            M = cv2.moments(cnt)
            compacidad.append()
            