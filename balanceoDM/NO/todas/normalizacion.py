# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 00:39:39 2019

@author: Nataly
"""

from pathlib import Path
datafolder=Path("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/txt")
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from yolovoc import yolo2voc
from readboxes import read_boxes
from rOI import ROI
from ventaneo import ventaneoo


import cv2
import numpy as np
import glob

from matplotlib import pyplot as plt


   
for image in glob.glob('*.jpg'):
    print(image)
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/NO/todas/'+image
    cv2.imwrite(dire,im)
    
    
    
    
   