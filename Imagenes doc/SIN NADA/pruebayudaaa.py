# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:34:17 2019

@author: Nataly
"""

from matplotlib import pyplot as plt
import glob
#from equalization import globalequalization, adaptativeequalization, contraststretching
import cv2
import numpy as np
img=cv2.imread('00000.jpg')
#img=cv2.imread('Lap_00598.jpg')

def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    #Adaptative Equalization
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)

    imha=cv2.merge((imhaR,imhaG,imhaB))
    
    return imha
def tloga(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img
img=cv2.resize(img,(500,500))
img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
img=adaptativeequalization(img)
#img=tloga(img)
R,G,B=cv2.split(img)

#img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
H,S,V=cv2.split(img)

#cv2.imshow('toda',img)
V = cv2.normalize(R, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

#cv2.imshow('R',R)
cv2.imshow('G',G)
#cv2.imshow('B',B)
cv2.waitKey(0)
#cv2.imshow('B',B)
#cv2.waitKey(0)
cv2.destroyAllWindows()
hist = cv2.calcHist([G],[0],None,[256],[0,255])
plt.plot(hist)
plt.show()
#umbral1=0.35*255
#umbral2=0.75*255
ta=G.shape
ta=list(ta)
binary=G.copy()
for f in range(ta[0]):
    for c in range (ta[1]):
        if G[f,c]>240:
            binary[f,c]=255
        else:
            binary[f,c]=0
cv2.imshow('B',binary)
cv2.waitKey(0)
cv2.destroyAllWindows()