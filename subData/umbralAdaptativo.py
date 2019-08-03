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
from skimage.filters.rank import minimum
from skimage.morphology import disk
from rOI import ROI
from normalizacion import log
from filMin import filtrominimo
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##

i=0
melocoton=0
z=0
import glob
for imgfile in glob.glob("*.jpg"):
    ima='./segROI/#5/Z3/'+imgfile
    imaROI=cv2.imread(ima,0)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    print(imgfile)
    
    img=cv2.imread(imgfile)
    img=log(img)
    img=filtrominimo(img)
   
    #UMBRAL ARTICULO
    #YUV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    YUV=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    for z in range(3):
        YUV[:,:,z]=YUV[:,:,z]*imaROI
    Y,U,V=cv2.split(YUV)    

    #Y = cv2.normalize(Y, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #U= cv2.normalize(U, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #V= cv2.normalize(V, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    #imaROI=imaROI*-1    
    #Y=Y*imaROI
    #U=U*imaROI
    #V=V*imaROI
    ta=V.shape
    ta=list(ta)
    MIN=int(np.min(V))
    MAX=int(np.max(V))
  
    hist = cv2.calcHist([V],[0],None,[MAX+1],[MIN,MAX])   #plt.plot(hist)
    #plt.plot(hist)
    #plt.show()
    umbral=200
    binary=V.copy()
    print(umbral)
    for f in range(ta[0]):
        for c in range (ta[1]):
            if V[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=255

    #binary = (binary*255).astype(np.uint8)
    #"""
    #cv2.imshow('image',Y)
    #cv2.waitKey(0)
    #cv2.imshow('image',U)
    #cv2.waitKey(0)
    #cv2.imshow('image',binary)
    #cv2.waitKey(0)
    #"""
    
    """#Intento 1_ Y>U#
    ta=U.shape
    ta=list(ta)
    binary=U.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if Y[f,c]>V[f,c]:
            #if V[f,c]<U[f,c]:
                binary[f,c]=0
            else:
                binary[f,c]=255
    binary=binary*imaROI
    #cv2.imshow('image',binary)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #"""
    
    #"""
    #binary=binary*-1
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('image',openi)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #15-15
    #25-25
    #openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dilatacion = cv2.dilate(binary,kernel,iterations = 1)
    #openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #"""
    
    """
    cv2.imshow('image',dilatacion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)
    """
    cv2.imshow('image',close)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #"""
    #"""
    #close=close*imaROI
    #close=close*-1
    #close= cv2.normalize(close, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #close = (close*255).astype(np.uint8)
    filetxt=imgfile[0:len(imgfile)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, img.shape)  
    re=0
    dm=0
    
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            dm=dm+1
        if cls==0:
            re=re+1
            
    if dm>0 and re==0:
        dire='./segmentacionRE2_combinacion/DM/'+imgfile
        cv2.imwrite(dire,close)
        print('dm')
    else:
        direM='./segmentacionRE_combinacion/RE/'+imgfile
        cv2.imwrite(direM,close)
        print('re')
    print(melocoton,z)
    i=i+1

    #"""
    #k = cv2.waitKey(1000)
    
    
    
    
    