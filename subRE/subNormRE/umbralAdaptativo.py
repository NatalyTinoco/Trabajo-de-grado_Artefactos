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
    img=cv2.imread(imgfile+'jpg')
    img=filtrominimo(img)
   
    #imgg=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)  
    #plt.imshow(imgg) 
    #plt.show()
    
    ima='./segROI/'+imgfile+'jpg'
    imaROI=cv2.imread(ima)
    imaROI=imaROI[:,:,2]
    #imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    cv2.imshow('image',imaROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    H,S,V=cv2.split(HSV)    
  
    
    H = cv2.normalize(H, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #S= cv2.normalize(S, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #V= cv2.normalize(V, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    imaROI=imaROI*-1
    plt.imshow(imaROI,cmap='Greys') 
    plt.show()
    H=H*imaROI
    S=S*imaROI
    V=V*imaROI
    #plt.imshow(S,'Greys') 
    #plt.show()
    plt.imshow(H,'Greys') 
    plt.show()
    
    #hist = cv2.calcHist([H],[0],None,[256],[0,255])
    #plt.plot(hist)
    #plt.show()
    
    plt.imshow(S,'Greys') 
    plt.show()
    #plt.imshow(V,'Greys') 
    #plt.show()
    #umbral=np.max(H)*0.9
    #print(umbral)
    ta=S.shape
    ta=list(ta)
    binary=H.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if H[f,c]<S[f,c]:
            #if S[f,c]<H[f,c]:
                binary[f,c]=0
            else:
                binary[f,c]=1
    plt.imshow(binary,cmap='Greys') 
    plt.show()
    
    """
    for f in range(ta[0]):
        for c in range (ta[1]):
            if H[f,c]>254:
                binary[f,c]=1
            else:
                binary[f,c]=0
    plt.imshow(binary,cmap='Greys') 
    plt.show()
    binary=S-V
    plt.imshow(S-V,cmap='Greys') 
    plt.show()
    """
    binary=binary*imaROI
    #binary=binary*-1
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    plt.imshow(openi,cmap='Greys') 
    plt.show()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close=cv2.morphologyEx(openi, cv2.MORPH_CLOSE, kernel)
    #close=close*imaROI
    close=close*-1
    #close= cv2.normalize(close, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #close = (close*255).astype(np.uint8)
    plt.imshow(close,cmap='Greys') 
    plt.show()
    dire='./segmentacionRE/'+imgfile+'jpg'
    cv2.imwrite(dire,close*255)
    k = cv2.waitKey(1000)
    
    """
    #print(np.mean(S),np.mean(V))
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