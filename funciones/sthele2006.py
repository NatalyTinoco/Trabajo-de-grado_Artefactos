# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:20:13 2019

@author: Usuario
"""

import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')

import cv2
import numpy as np 
from rOI import ROI
    
def sthele2006(imgfile):
    #    imgfile='00026_batch2.jpg'
    #imgFile = 'Lap_00000.jpg'
    img=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/'+imgfile)  
    imaROI=ROI(img)
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    img=adaptativeequalization(img)    
<<<<<<< HEAD
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
=======
    imaROI = cv2.normalize(imaROI, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
>>>>>>> c01490c95d56aa6eaa0946d849dfbfba46fe4037
    for z in range(3):
        img[:,:,z]=img[:,:,z]*imaROI
 
    #_,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(imaROI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    
#    cv2.imshow('image',imaROI)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows() 
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    R,G,B=cv2.split(img)    
    V=G[y3:y3+h3,x3:x3+w3]
    
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
    #print(umbral)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilatacion = cv2.dilate( Binaryfinal,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)

    return close

    
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
