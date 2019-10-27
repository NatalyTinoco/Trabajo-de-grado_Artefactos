# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 00:46:03 2019

@author: Nataly
"""

import sys
sys.path.insert(1, 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from test_todoRE import test_all_RE
from test_todoDM import test_all_DM
#from readimg import read_img
import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
import math

def mse(x,y):
    a = np.array(x) # your x
    b = np.array(y) # your y
    mses = np.mean(((a-b)**2))
    return mses
msedm=[],[],[]
msere=[],[],[]
for image in glob.glob('*.jpg'):    
    im = cv2.imread(image)
    filetxt='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/Todas/'+image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    
    resul, original_2,imDU_2,umbrImage,original_3,bboxre=test_all_RE(image)
    resuldm,originaldm_2,imDRdm_2,original_3,bboxdm=test_all_DM(image,original_3)
    
    print(bboxre[0],bboxre[1],bboxre[2],bboxre[3],bboxre[4])
    print(bboxdm[0],bboxdm[1],bboxdm[2],bboxdm[3],bboxdm[4])
    
    re=0
    dm=0
  
   
    for b in boxes_abs:
            cls, x1, y1, x2, y2 = b            
            a=[int(x1),int(y1),int(x2),int(y2)]
            if cls == 3:
                b=[bboxdm[1],bboxdm[2],bboxdm[3],bboxdm[4]]
                print(0,x1,y1,x2,y2)
                dm=dm+1
                print('DM')
                menordm=[]
                if len(bboxdm[0])>0:
                    for j in range(len(bboxdm[0])):
                        b=[bboxdm[1][j],bboxdm[2][j],bboxdm[3][j],bboxdm[4][j]]
                        menordm.append(mse(a,b))
                    uu=menordm.index(np.min(menordm))
                    msedm[0].append(a)
                    msedm[1].append([bboxdm[1][uu],bboxdm[1][uu],bboxdm[3][uu],bboxdm[4][uu]])
                    msedm[2].append(np.min(menordm))
                else:
                    msedm[0].append(a)
                    msedm[1].append('nan')
                    msedm[2].append('nan')
                    
            if cls==0:
                print(1,x1,y1,x2,y2)
                menorre=[]
                re=re+1
                print('RE')
                if len(bboxre)>0:
                    for j in range(len(bboxre[0])):
                        b=[bboxre[1][j],bboxre[2][j],bboxre[3][j],bboxre[4][j]]
                        menorre.append(mse(a,b))
                        print('============', mse(a,b))
                    u=menorre.index(np.min(menorre))
                    msere[0].append(a)
                    msere[1].append([bboxre[1][u],bboxre[2][u],bboxre[3][u],bboxre[4][u]])
                    msere[2].append(np.min(menorre))
                else:
                    msere[0].append(a)
                    msere[1].append('nan')
                    msere[2].append('nan')
          

import pandas as pd               
datos = {'Anotación BD': msere[0][:],
         'Ubicación método propuesto':msere[1][:],
         'MSE':msere[2][:]}
conso=pd.DataFrame(datos)
conso.to_excel('EvaluacionubicacionRE.xlsx')

datos = {'Anotación BD': msedm[0][:],
         'Ubicación método propuesto':msedm[1][:],
         'MSE':msedm[2][:]}
conso=pd.DataFrame(datos)
conso.to_excel('EvaluacionubicacionDM.xlsx')                
                
                
                