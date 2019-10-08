# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:28:45 2019

@author: Usuario
"""

import cv2
import numpy as np 
import pylab as plt 


def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def ventanIDEA(imDR,imaROI):   
#    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(imaROI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)

    R,G,B=cv2.split(imDR)
    V=G[y3:y3+h3,x3:x3+w3]
    Binary=V.copy()
    a,b = V.shape

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
            gradiente=np.gradient(zz[200:])
            uu=find_nearest(gradiente,0)
            gradiente=gradiente.tolist()
            umbral1 = gradiente.index(uu)
            umbral=200+umbral1
            
            hl=hl+1
            
            ta1,ta2=cropped.shape
            binary=cropped.copy()
            for ff in range(ta1):
                   for cc in range (ta2):
                       if cropped[ff,cc]<umbral:
                           binary[ff,cc]=0
                       else:
                           binary[ff,cc]=255
            if c<tamañoB*vecesB-tamañoB and f<tamañoA*vecesA-tamañoA:
               Binary[f:f+tamañoA,c:c+tamañoB] = binary
            if c==tamañoB*vecesB-tamañoB and f<tamañoA*vecesA-tamañoA:
               Binary[f:f+tamañoA,c:]=binary
            if f==tamañoA*vecesA-tamañoA:
               if c==tamañoB*vecesB-tamañoB:
                  Binary[f:,c:]=binary
               else: 
                  Binary[f:,c:c+tamañoB]=binary
 
    fila,Col=G.shape
    Binaryfinal=np.zeros((fila,Col)).astype(np.uint8)
    Binaryfinal[y3:y3+h3,x3:x3+w3]=Binary    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilatacion = cv2.dilate( Binaryfinal,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)
    
    return close

