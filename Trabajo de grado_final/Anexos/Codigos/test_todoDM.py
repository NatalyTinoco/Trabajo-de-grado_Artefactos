# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:53:35 2019

@author: Daniela y Nataly 

Descripción del código: Código donde se une toda la metodología para identificación de desenfoque por movimiento (mDM).
*Entrada: imagePath1(ruta de la imagen) y original_3 (salida de la mRE)
*Salida: predictions (predicción del clasificador (1-Otros y 0-DM)), original_2 (imagen original), imDR_2(Imagen sin etiquetas), original_5 (imagen con bbox graficados, ubicación de los artefactos)
y bboxdm (posición de los bbox).

"""
import cv2
import pickle
import pandas as pd
from Normalizacion import normalizacionMaxMin
from rOI import ROI
from caracDM import carcDM
from ventaneo import ventaneoo,ventadibujo
import numpy as np

with open('model_pickle_DM','rb') as f:
    mpDM = pickle.load(f)
def contorn(roiImage_1):
    try:
        contours,hierachy = cv2.findContours(roiImage_1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:    
        _,contours,_= cv2.findContours(roiImage_1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x1,y1,w1,h1 = cv2.boundingRect(cnt) 
    return x1,y1,w1,h1

def test_all_DM(imagePath1,original_3):
    original = cv2.imread(imagePath1)
    xx,yy,cc=original.shape       
    original_2 = original.copy()
    original_4=original_3.copy()
    original_5= original.copy()
    original_3_1=original_3.copy()
    imNorm = normalizacionMaxMin(original)
    imDR = imNorm.copy()
    roiImage = ROI(imNorm)
    for z in range(3):
        imDR[:,:,z]=imNorm[:,:,z]*roiImage
    imDR_2=imDR.copy()
    for z in range(3):
        imDR_2[:,:,z]=original_2[:,:,z]*roiImage
    hh,ww,cc=imDR.shape
    roiImage_1=roiImage.copy()
    upa=320
    apa=520
    if hh<upa and ww< upa :
       imDR=cv2.resize(imDR,(apa,apa))   
       roiImage=cv2.resize(roiImage,(apa,apa)) 
       original_3_1=cv2.resize(original_3_1,(apa,apa)) 
    if hh<upa and ww>upa:
       imDR=cv2.resize(imDR,(apa,ww))   
       roiImage=cv2.resize(roiImage,(apa,ww)) 
       original_3_1=cv2.resize(original_3_1,(apa,ww)) 
    if ww<upa and hh>upa:
       imDR=cv2.resize(imDR,(hh,apa))   
       roiImage=cv2.resize(roiImage,(hh,apa)) 
       original_3_1=cv2.resize(original_3_1,(hh,apa)) 
    x,y,w,h = contorn(roiImage)
    otra=imDR[y:y+h,x:x+w]
    a,b,ch =otra.shape
    tamañoa1A=300
    tamañoa1B=300
    predictions = []
    bboxdm=[],[],[],[],[]
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
        for ca1 in range(0,b-tamañoa1B,tamañoa1B):       
            croppeda1=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, otra)
            entropia,ssimn=carcDM(croppeda1)
            carac=pd.DataFrame({'entropia':entropia,'ssimn':ssimn},index =['1'])
            pred=int(mpDM.predict(carac))
            predictions.append(pred)
            if pred == 0:
               original_3_1=ventadibujo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, original_3_1,bboxdm)
    if np.mean(predictions)==0:  
        x1,y1,w1,h1 = contorn(roiImage_1)
        cv2.rectangle(original_4,(x1,y1),(int(x1+w1-5),int(y1+h1-5)),(255,0,0),2)   
        original_5=original_4
        bboxdm[0].append(0)
        bboxdm[1].append(int(x1))
        bboxdm[2].append(int(y1))
        bboxdm[3].append(int(w1))
        bboxdm[4].append(int(h1))
    else:
        original_5=cv2.resize(original_3_1,(yy,xx))       
    return predictions,original_2,imDR_2,original_5,bboxdm

