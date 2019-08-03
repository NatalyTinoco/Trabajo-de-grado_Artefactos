# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:50:34 2019

@author: Nataly
"""

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
from pylab import *
import glob
import xlrd
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li
from skimage.filters import threshold_minimum
workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

#for imgfile in glob.glob("*.jpg"):
for col in range(sheet.ncols):
    #imgfile='00000.'
    imgfile = sheet.cell_value(0, col)  
    imgfile=imgfile+'jpg'
    print(imgfile)
    imaY=cv2.imread("./Espaciodecolor/Y/"+imgfile,0)
    imaV=cv2.imread("./Espaciodecolor/V/"+imgfile,0)
    plt.imshow(imaY,cmap=plt.cm.gray)
    plt.show()
    plt.imshow(imaV,cmap=plt.cm.gray)
    plt.show()
    #histY = cv2.calcHist([imaY],[0],None,[256],[1,255])
    #plt.plot(histY)
    #plt.show()
    #histV = cv2.calcHist([imaV],[0],None,[256],[0,255])
    #plt.plot(histV)
    #plt.show()
    
    ta=imaY.shape
    ta=list(ta)
    segmenta=imaY.copy()

    ta=imaY.shape
    ta=list(ta)
    segmenta=imaY.copy()
    for f in range (ta[0]):
        for c in range(ta[1]):
            if imaY[f,c]>imaV[f,c]:
                segmenta[f,c]=1
            else:
                segmenta[f,c]=0
    plt.imshow(segmenta,'Greys')
    plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #openi = cv2.morphologyEx(segmenta, cv2.MORPH_OPEN, kernel)
    dilata=cv2.dilate(segmenta,kernel,1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close=cv2.morphologyEx(dilata, cv2.MORPH_CLOSE, kernel)
    plt.imshow(close,'Greys')
    plt.show()

            