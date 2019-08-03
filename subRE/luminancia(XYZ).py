# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:28:06 2019

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

def log(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img
    
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
    img=log(img)
    img=filtrominimo(img)
   
    #imgg=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)  
    #plt.imshow(imgg) 
    #plt.show()
    
    ima='./subNormRE/segROI/'+imgfile+'jpg'
    imaROI=cv2.imread(ima,0)
    #imaROI=imaROI[:,:,2]
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    #cv2.imshow('image',imaROI)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    XYZ=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
   # YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    X,Y,Z=cv2.split(XYZ)    

    #Y = cv2.normalize(Y, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #U= cv2.normalize(U, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #V= cv2.normalize(V, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    #imaROI=imaROI*-1    
    X=X*imaROI
    Y=Y*imaROI
    Z=Z*imaROI
    #"""
    cv2.imshow('image',X)
    cv2.waitKey(0)
    cv2.imshow('image',Z)
    cv2.waitKey(0)
    #"""
    
    y=Y/(X+Y+Z)
    cv2.imshow('image',Y)
    cv2.waitKey(0)
    cv2.imshow('image',y)
    cv2.waitKey(0)
    #"""#Intento 1_ Y>U#
    ta=X.shape
    ta=list(ta)
    binary=X.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if Y[f,c]>Z[f,c]:
            #if S[f,c]<H[f,c]:
                binary[f,c]=255
            else:
                binary[f,c]=0
    binary=binary*imaROI
    cv2.imshow('image',binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #"""
    
   
    """#Umbrales articulo#
    #print(np.mean(S),np.mean(V))
    umbral1=0.35*255
    umbral2=0.75*255
    ta=V.shape
    ta=list(ta)
    binary=V.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if U[f,c]<umbral1 and V[f,c]>umbral2:
                binary[f,c]=255
            else:
                binary[f,c]=0
    binary=binary*imaROI
    #cv2.imshow('image',binary)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #"""
    
    #binary=binary*-1
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('image',openi)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #15-20
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilatacion = cv2.dilate(binary,kernel,iterations = 1)
    #openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('image',dilatacion)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('image',close)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #close=close*imaROI
    #close=close*-1
    #close= cv2.normalize(close, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #close = (close*255).astype(np.uint8)


    dire='./subNormRE/segmentacionRE/'+imgfile+'jpg'
    #cv2.imwrite(dire,close)
    k = cv2.waitKey(1000)