# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:12:57 2019

@author: Nataly
"""

import cv2
import numpy as np
import pylab as plt 
from matplotlib import pyplot as plt
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr    
import xlrd
from skimage.filters.rank import minimum
from skimage.morphology import disk

def filtrominimo(img):
    B,G,R=cv2.split(img)
    fR = minimum(R, disk(3))
    fG = minimum(G, disk(3))
    fB = minimum(B, disk(3))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

for col in range(sheet.ncols):
    imgfile = sheet.cell_value(0, col)  
    print(imgfile)
    #imgf='./subNormRE/FiltroRE/Espaciodecolor/V/'+imgfile+'jpg'
    img=cv2.imread(imgfile+'jpg')
    img=filtrominimo(img)
    
    #img=cv2.imread('00000.jpg')
    imgg=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)  
    plt.imshow(imgg) 
    plt.show()
    
    ima='./segROI/'+imgfile+'jpg'
    imaROI=cv2.imread(ima)
    
    plt.imshow(imaROI) 
    plt.show()
    
    
    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    H,S,V=cv2.split(HSV)
    H=H*imaROI
    S=S*imaROI
    V=V*imaROI
    
    plt.imshow(S,'Greys') 
    plt.show()
    
    H = cv2.normalize(H, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    S= cv2.normalize(S, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #V= cv2.normalize(V, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    plt.imshow(H,'Greys') 
    plt.show()
    hist = cv2.calcHist([H],[0],None,[256],[0,255])
    plt.plot(hist)
    plt.show()
    #plt.imshow(S,'Greys') 
    #plt.show()
    #plt.imshow(V,'Greys') 
    #plt.show()
    umbral=np.max(H)*0.9
    print(umbral)
    ta=S.shape
    ta=list(ta)
    binary=H.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if H[f,c]>254:
                binary[f,c]=1
            else:
                binary[f,c]=0
    plt.imshow(binary,cmap='Greys') 
    plt.show()
    
    """binary=S-V
    plt.imshow(S-V,cmap='Greys') 
    plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    close=cv2.morphologyEx(openi, cv2.MORPH_CLOSE, kernel)
    plt.imshow(close,cmap='Greys') 
    plt.show()
    """
    #print(np.mean(S),np.mean(V))
    """
    ta=S.shape
    ta=list(ta)
    binary=H.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if S[f,c]<0.35 and V[f,c]>0.75:
                binary[f,c]=0
            else:
                binary[f,c]=1
    plt.imshow(binary,cmap='Greys') 
    plt.show()
    #"""