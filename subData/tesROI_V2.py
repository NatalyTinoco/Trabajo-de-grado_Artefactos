# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:54:25 2019

@author: Usuario
"""

import glob 
import cv2
import rOI

for imgfile in glob.glob("*.jpg"):
    ima=cv2.imread(imgfile)
    imaRoi=rOI.ROI(ima) 
    
    dire='./segROI/#5/Z1/'+imgfile
    cv2.imwrite(dire,imaRoi)
