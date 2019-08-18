# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 00:35:28 2019

@author: Nataly
"""

from pathlib import Path
datafolder=Path("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/txt")
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS')

#from readimg import read_img
import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt
from rOI import ROI

#tamañoA = []
#tamañoB = []
import shutil, os
i=13
for image in glob.glob('*.jpg'):
    im = cv2.imread(image)
    imgq=552
    imgcar=7450
    saltos=int(imgcar/imgq)
    largo=saltos*imgq
#    for i in range(imgq):
    if i%saltos==0:
        dire=dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/DM/'+image
        shutil.move(image,dire)
#        cv2.imwrite(dire,im)
#    numero=(i/saltos)+1
    i=i+13
    
    
    
    
    
    
    