# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:16:44 2019

@author: Nataly
"""

import cv2
import glob 
from matplotlib import pyplot as plt
for imgfile in glob.glob("*.jpg"):   
    imgseg=cv2.imread(imgfile,0)
    imgseg = cv2.normalize(imgseg, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    imgroi=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segROI/ROI_Manuales_V2/'+imgfile,0)
    imgroi = cv2.normalize(imgroi, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#    plt.imshow(imgroi,'Greys')
#    plt.show()
#    plt.imshow(imgseg,'Greys')
#    plt.show()
    inueva=imgseg *imgroi
#    plt.imshow(inueva,'Greys')
#    plt.show()
    inueva = cv2.normalize(inueva, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subDM/segGT/'+imgfile
    cv2.imwrite(dire,inueva)
#    