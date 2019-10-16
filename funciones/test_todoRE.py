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
    
with open('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/Método de identificación/model_pickle','rb') as f:
    mpRE = pickle.load(f)
   
def test_all_RE(imagePath):
    original = cv2.imread(imagePath)
    imNorm = normalizacionMaxMin(original)
    imEqu = adaptativeequalization(imNorm)
    imDR = imEqu.copy()
    roiImage = ROI(imEqu)
    for z in range(3):
        imDR[:,:,z]=imEqu[:,:,z]*roiImage
    imDU = imDR.copy()
    umbrImage = ventanIDEA(imDR,roiImage)
    for z in range(3):
        imDU[:,:,z]=imDR[:,:,z]*umbrImage
        
    contours,hierachy = cv2.findContours(umbrImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # _,contours,_ = cv2.findContours(close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    predict=[]
    for c in range(len(contours)):
        cnt = contours[c]
#        epsilon = 0.01*cv2.arcLength(cnt,True)
#        approx = cv2.approxPolyDP(cnt,epsilon,True)
        x,y,w,h = cv2.boundingRect(cnt)
        
#        cropped1 = imDU[int(y):int(y+h),int(x):int(x+w)]
        cropped2 = imDR[int(y):int(y+h),int(x):int(x+w)]
        
        brillo,contraste,desvi=caracRe(cropped2)
        carac=pd.DataFrame({'contrastB':contraste,'desviacionB':desvi,'Brillo':brillo},index =['1'])
        pred=int(mpRE.predict(carac))
        predict.append(pred)
#        print(pred)
        
        if pred == 1:
            umbrImage[int(y):int(y+h),int(x):int(x+w)] = umbrImage[int(y):int(y+h),int(x):int(x+w)]
        else:
            umbrImage[int(y):int(y+h),int(x):int(x+w)] = 0
            
    return predict
#    cv2.imwrite('contorno.jpg',original)
#    cv2.imwrite('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/test-todo/'+filePath+'/'+str(c)+'-contorno.jpg',cropped)

    
#correccion1=suavizado(original,umbrImage,15)
#correccion2=inpaintingB(original,umbrImage)
#correccion3=inpaintingNS(original,umbrImage)
#correccion4=inpaintingTA(original,umbrImage)
    
   

#cv2.imshow('imageres', roiImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#im_rgb = cv2.cvtColor(umbrImage, cv2.COLOR_BGR2RGB)
#plt.imshow(im_rgb)
#plt.show()
