# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:43:43 2019

@author: Nataly
"""
import sys
path = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/funciones' 
#path= 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones'
sys.path.insert(1,path)
from test_todoRE import test_all_RE
from test_todoDM import test_all_DM
import numpy as np
import math
import glob
import cv2
#    filetxt=file[0:len(file)-3]+'txt'      
#    bboxfile=filetxt
xtest=[]
xpre=[]
for file in glob.glob("*.jpg"):
    imgfile=file
    img = cv2.imread(file)
    resul, original_2,imDU_2,umbrImage,original_3=test_all_RE(file)
    resuldm,originaldm_2,imDRdm_2,original_3=test_all_DM(file,original_3)
    
    grupo0 = [i for i,x in enumerate(resul) if x == 0]
    grupo1 = [i for i,x in enumerate(resul) if x == 1]
    grupodm0 = [i for i,x in enumerate(resuldm) if x == 0]
    grupodm1 = [i for i,x in enumerate(resuldm) if x == 1]
    xtest.append(1)
        
    if len(grupo1)>0 and len(grupodm0)==0:
        cv2.imwrite('./clasifica/'+file,img)
        xpre.append(1)
    if len(grupodm0)>0 and len(grupo1)==0:
        xpre.append(0)
    if  len(grupodm0)>0 and len(grupo1)>0 :
        xpre.append(3)     
    if len(grupo0)>0 and len(grupodm1)>0 and len(grupo1)==0 and len(grupodm0)==0 or math.isnan(np.mean(resul))==True and len(grupodm1)>0 and len(grupodm0)==0:
        xpre.append(2) 
        
import pandas as pd  
datos = {'Y': xtest,
         'predic':xpre}

datos = pd.DataFrame(datos)
#med 240
#med_2 250

datos.to_excel('Med_re_4.xlsx')      