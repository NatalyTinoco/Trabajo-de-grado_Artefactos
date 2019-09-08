# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:28:45 2019

@author: Usuario
"""
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from rOI import ROI


import cv2
import numpy as np 
import pylab as plt 
from matplotlib import pyplot as plt
from skimage import exposure
from scipy.signal import find_peaks


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


def ventanIDEA(imgfile):   
#    img=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/'+imgfile)   
    img=imgfile
    imaROI=ROI(img)
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    img=adaptativeequalization(img)
    
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range(3):
        img[:,:,z]=img[:,:,z]*imaROI
        
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#    contours,hierarchy = cv2.findContours(imaROI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)

    R,G,B=cv2.split(img)
    V=G[y3:y3+h3,x3:x3+w3]
    original=G.copy()
    Binary=V.copy()
    a,b = V.shape
#    tamañoA = 150
#    tamañoB = 150
#    tamañoA = 50
#    tamañoB = 50
#    tamañoA = 256
#    tamañoB = 256
    tamañoA = 200
    tamañoB = 200
    
    vecesA = int(a/tamañoA)
    vecesB = int(b/tamañoB)
    hl=0

    for f in range(0,a-tamañoA,tamañoA):
        for c in range(0,b-tamañoB,tamañoB):
            cropped = V[f:f+tamañoA,c:c+tamañoB]
            if c==tamañoB*vecesB-tamañoB:
                cropped = V[f:f+tamañoA,c:]
            if f==tamañoA*vecesA-tamañoA:
                 if c==tamañoB*vecesB-tamañoB:
                    cropped = V[f:,c:]
                 else:
                     cropped = V[f:,c:c+tamañoB]
            ta=cropped.shape
            ta=list(ta)
            binary=cropped.copy()
            hist = cv2.calcHist([cropped],[0],None,[256],[0,255])
            hist=np.asarray(hist).astype(np.int)
            zz=list(range(0,len(hist)))
            for ii in range(len(hist)):
                zz[ii]=int(hist[ii])
 
            hist=hist.tolist() 
            plt.plot(hist)
            plt.show()
            gradiente=np.gradient(zz[200:])
            uu=find_nearest(gradiente,0)
            gradiente=gradiente.tolist()
            umbral1 = gradiente.index(uu)
            umbral=200+umbral1
            #print('ventana',hl)
            hl=hl+1
            #print('UMBRAL=',umbral)
            ta1,ta2=cropped.shape
            binary=cropped.copy()
            for ff in range(ta1):
                   for cc in range (ta2):
                       if cropped[ff,cc]<umbral:
                           binary[ff,cc]=0
                       else:
                           binary[ff,cc]=255
            if c<tamañoB*vecesB-tamañoB and f<tamañoA*vecesA-tamañoA:
#               cv2.rectangle(original,(f,c),(f+tamañoA,c+tamañoB),(255,0,0),2)
               Binary[f:f+tamañoA,c:c+tamañoB] = binary
            if c==tamañoB*vecesB-tamañoB and f<tamañoA*vecesA-tamañoA:
#               cv2.rectangle(original,(f,c),(f+tamañoA,c+b),(255,0,0),2) 
               Binary[f:f+tamañoA,c:]=binary
            if f==tamañoA*vecesA-tamañoA:
               if c==tamañoB*vecesB-tamañoB:
#                  cv2.rectangle(original,(f,c),(f+a,c+b),(255,0,0),2) 
                  Binary[f:,c:]=binary
               else:
#                  cv2.rectangle(original,(f,c),(f+a,c+tamañoB),(255,0,0),2) 
                  Binary[f:,c:c+tamañoB]=binary
#  #%%                
#    original=cv2.resize(original,(900,800))
#    cv2.imshow('image',original)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()     
    fila,Col=G.shape
    Binaryfinal=np.zeros((fila,Col)).astype(np.uint8)
    Binaryfinal[y3:y3+h3,x3:x3+w3]=Binary    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilatacion = cv2.dilate( Binaryfinal,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)
    
    return close
#    original=cv2.resize(close,(600,500))
#    cv2.imshow('image',original)
#    cv2.waitKey(0)
