# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:53:35 2019

@author: Usuario
"""
import cv2
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from Normalizacion import normalizacionMaxMin
from equalization import adaptativeequalization
from rOI import ROI
from ventanIDEA import ventanIDEA
from caracRE import caracRe
#from correccion import suavizado,inpaintingB,inpaintingNS,inpaintingTA

#imagePath = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE/00095.jpg'
  
fileOpen = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/Método de identificación/model_pickle'  
#fileOpen = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/model_pickle'

with open(fileOpen,'rb') as f:
    mpRE = pickle.load(f)
   
def test_all_RE(imagePath):
    original = cv2.imread(imagePath)
    original_2 =original.copy() 
    original_3=original.copy() 
    imNorm = normalizacionMaxMin(original)
    imEqu = adaptativeequalization(imNorm)
    imDR = imEqu.copy()
    roiImage = ROI(imEqu)
    for z in range(3):
        imDR[:,:,z]=imEqu[:,:,z]*roiImage
    
    imDR_2=original_2.copy()
    for z in range(3):
        imDR_2[:,:,z]=original_2[:,:,z]*roiImage
   
    imDU_2=imDR_2.copy()
    imDU = imDR.copy()
    umbrImage = ventanIDEA(imDR,roiImage)
    for z in range(3):
        imDU[:,:,z]=imDR[:,:,z]*umbrImage
        
        
    try:
        contours,hierachy = cv2.findContours(umbrImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:    
        _,contours,_ = cv2.findContours(umbrImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
    predic=[]

    if len(contours)==0:
         pred=0
    else:
        for c in range(len(contours)):
            cnt = contours[c]
            x,y,w,h = cv2.boundingRect(cnt)
            
            cropped2 = imDR[int(y):int(y+h),int(x):int(x+w)]
#            cropped2 = imDU[int(y):int(y+h),int(x):int(x+w)]
            
            brillo,contraste,desvi=caracRe(cropped2)
            carac=pd.DataFrame({'contrastB':contraste,'desviacionB':desvi,'Brillo':brillo},index =['1'])
            pred=int(mpRE.predict(carac))
            
            if pred == 1:
                umbrImage[int(y):int(y+h),int(x):int(x+w)] = umbrImage[int(y):int(y+h),int(x):int(x+w)]
                cv2.rectangle(original_3,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),2)  
            else:
                umbrImage[int(y):int(y+h),int(x):int(x+w)] = 0
            
            predic.append(pred)
            
    return predic, original_2 , imDU_2,umbrImage,original_3

