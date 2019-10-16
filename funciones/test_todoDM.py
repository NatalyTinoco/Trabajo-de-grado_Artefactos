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
from rOI import ROI
from caracDM import carcDM
from ventaneo import ventaneoo
import numpy as np

#imagePath1 = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE/00000.jpg'
#with open('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/test-todo/model_pickle_DM','rb') as f:
with open('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/test-todo/model_pickle_DM','rb') as f:
    mpDM = pickle.load(f)

def test_all_DM(imagePath1):
    original = cv2.imread(imagePath1)
    original_2 = original.copy()
    imNorm = normalizacionMaxMin(original)
    imDR = imNorm.copy()
    roiImage = ROI(imNorm)
    for z in range(3):
        imDR[:,:,z]=imNorm[:,:,z]*roiImage
    
    imDR_2=imDR.copy()
    for z in range(3):
        imDR_2[:,:,z]=original_2[:,:,z]*roiImage
        
    print(imDR.shape)
    hh,ww,cc=imDR.shape
    if hh<310 and ww<310:
       imDR=cv2.resize(imDR,(500,500))   
       roiImage=cv2.resize(roiImage,(500,500)) 
    if hh<310 and ww>310:
       imDR=cv2.resize(imDR,(500,ww))   
       roiImage=cv2.resize(roiImage,(500,ww)) 
    if ww<310 and hh>310:
       imDR=cv2.resize(imDR,(hh,500))   
       roiImage=cv2.resize(roiImage,(hh,500)) 
    
    _,contours,_= cv2.findContours(roiImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)    

    otra=imDR[y:y+h,x:x+w]
    a,b,ch =otra.shape
    
    tamañoa1A=300
    tamañoa1B=300
    
    predictions = []
    
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
        for ca1 in range(0,b-tamañoa1B,tamañoa1B):
            croppeda1=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, otra)
#            plt.imshow(cv2.cvtColor(croppeda1, cv2.COLOR_RGB2BGR))
#            plt.show()
            entropia,ssimn=carcDM(croppeda1)
            carac=pd.DataFrame({'entropia':entropia,'ssimn':ssimn},index =['1'])
            pred=int(mpDM.predict(carac))
            
            predictions.append(pred)
            
    return predictions,original_2,imDR_2

#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subDM/WL_00444.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/0000572.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/00019.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/CT56_colitis_05604.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/Lap_00647.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/gastro_00048.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/Flu_01038.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/FL-WL_00107.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/00323.jpg'
#imagePath1 = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS/00090_batch2.jpg'


#imagePath1 ='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/RE/AN12_344.jpg'
#testing=test_all_DM(imagePath1)   
#print('==========',testing)    
#cv2.imshow('imagen0', imacropped[2])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

