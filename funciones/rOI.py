# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:22:47 2019

@author: Nataly
"""
import cv2
import numpy as np
def ROI(ima):
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2XYZ)
    I,I,II=cv2.split(imA)
    
    ta=II.shape
    ta=list(ta)
    hist = cv2.calcHist([II],[0],None,[256],[0,255])
    
    div=6
    
    nuevohist=hist.tolist() 
    l=int(len(nuevohist)/div)
    i=1
    a=0
    suma=np.zeros((div))
    valm=0
    hasta=0
    for y in range(1,div+1):
        suma[a]=int(sum(np.asarray(nuevohist[i:int(l)*y])))
        if suma[a]>valm:
            valm=suma[a]
            hasta=y
        i=int(l)*y+1
        a=a+1

    
    porcen=0.3
 
    a=hist[:int((l*hasta)*porcen)]
    a=a.tolist()         
    hist=hist.tolist() 
    u=np.max(a)
    umbral1 = hist.index(u)
    a=a[umbral1:]
    
    uu=np.min(a)
    histb=hist[umbral1:]
    
    umbral=histb.index(uu)
    
    umbral=umbral+len(hist[:umbral1])
    
    binary=II.copy()
#    print(umbral)
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=1

    binary = (binary).astype(np.uint8)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    close=cv2.morphologyEx(openi, cv2.MORPH_CLOSE, kernel)
    
#    contours,hierachy = cv2.findContours(close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    
    try:
        contours,hierachy = cv2.findContours(close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:    
        _,contours,_ = cv2.findContours(close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)

    tru=15
    for f in range ((y+tru),(y+h-tru)):
        for c in range(x+tru,(x+w-tru)):
            #print(f,c)
            if close[f+tru,c+tru]==1 or close[f+tru,c]==1 or close[f,c+tru]==1 and close[f,c]==0:
                close[f,c]=1
            else:
                close[f,c]=close[f,c]
                
    return close 