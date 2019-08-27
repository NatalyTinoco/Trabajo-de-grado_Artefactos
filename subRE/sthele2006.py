# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:31:20 2019

@author: Nataly
"""
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from yolovoc import yolo2voc
from readboxes import read_boxes
from rOI import ROI

import glob
import cv2
import numpy as np 
from rOI import ROI
from skimage import exposure
import pylab as plt 
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    #Adaptative Equalization
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)

    imha=cv2.merge((imhaR,imhaG,imhaB))
    
    return imha

def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]

for imgfile in glob.glob("*.jpg"):
    
#    imgfile='00026_batch2.jpg'
    img=cv2.imread(imgfile)  
#    original=img.copy()
#    original=cv2.resize(original,(600,500))
#    cv2.imshow('Original',original)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    img=adaptativeequalization(img)    
    
    imaROI=ROI(img)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range(3):
        img[:,:,z]=img[:,:,z]*imaROI

    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    R,G,B=cv2.split(img)    
    V=G[y3:y3+h3,x3:x3+w3]
    
#    original=G.copy()
#    original=cv2.resize(original,(600,500))
#    cv2.imshow('G',original)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    
    Binary=V.copy()
    hist = cv2.calcHist([V],[0],None,[256],[0,255])
    hist=np.asarray(hist).astype(np.int)
    zz=list(range(0,len(hist)))
    for ii in range(len(hist)):
        zz[ii]=int(hist[ii])
#    plt.plot(hist)    
#    plt.show()
#    plt.hist(V.ravel(),256,[0,256])
#    plt.show()
    gradiente=np.gradient(zz[200:])   
    uu=find_nearest(gradiente,0)
    gradiente=gradiente.tolist()
    umbral1 = gradiente.index(uu)
    umbral=200+umbral1
    print(umbral)
    ta1,ta2=V.shape
    Binary=V.copy()
    for ff in range(ta1):
           for cc in range (ta2):
               if V[ff,cc]<umbral:
                   Binary[ff,cc]=0
               else:
                   Binary[ff,cc]=255
    
    fila,Col=G.shape
    Binaryfinal=np.zeros((fila,Col)).astype(np.uint8)
    Binaryfinal[y3:y3+h3,x3:x3+w3]=Binary   
    #2-3
    #6-10
    #15-25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilatacion = cv2.dilate( Binaryfinal,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)
    
#    original=close.copy()
#    original=cv2.resize(original,(600,500))
#    cv2.imshow('close',original)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    
    

