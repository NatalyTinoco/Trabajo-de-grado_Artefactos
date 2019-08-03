# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:38:01 2019

@author: Nataly
"""
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

for imgfile in glob.glob("*.jpg"):
    original = cv2.imread(imgfile)
    
    II = cv2.normalize(original, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
     #II=original[:,:,1]*-1
    II= II.astype(np.uint8)
    #plt.imshow(II,'Greys')
    #plt.show() 
    #hist = cv2.calcHist([II],[0],None,[256],[0,255])
    #plt.plot(hist)
    #plt.show()
    dire='./'+imgfile
    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)  
    cv2.imwrite(dire,II)