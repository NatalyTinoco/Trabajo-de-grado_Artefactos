# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:36:19 2019

@author: Usuario
"""

#from readimg import read_img
import cv2
import numpy as np
import glob

tamañoA = []
tamañoB = []

for image in glob.glob('*.jpg'):
    img = cv2.imread(image)
    a,b,c = img.shape
    tamañoA.append(a)
    tamañoB.append(b)
    
    for y in range(a):
        for x in range(b):
            

#promedioa = np.mean(tamañoA)
#promediob = np.mean(tamañoB)
