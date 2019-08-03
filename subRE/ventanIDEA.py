# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:51:59 2019

@author: Nataly
"""

import glob
import cv2
import numpy as np 
from rOI import ROI
from normalizacion import log
from filMin import filtrominimo
for imgfile in glob.glob("*.jpg"):
    ima='./segROI/Z3/'+imgfile+'jpg'
    imaROI=cv2.imread(ima,0)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    img=cv2.imread(imgfile+'jpg')
    img=log(img)
    #img=filtrominimo(img)
    YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y,U,V=cv2.split(YUV)   
    cv2.imshow('image',V)
    cv2.waitKey(0)
    cv2.destroyAllWindows()