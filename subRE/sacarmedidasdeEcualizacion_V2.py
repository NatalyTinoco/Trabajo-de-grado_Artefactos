# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:05:52 2019

@author: Usuario
"""

#from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from comparacionhistReflejosE import comparacionhistRE

from equalization import globalequalization, adaptativeequalization, contraststretching
from Normalizacion import  normalizacionMaxMin

import cv2
import xlrd
from matplotlib import pyplot as plt

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

c=0
a=1
r=0
i=0

correlacionT = [[],[],[],[]]
BhattacharyyaT = [[],[],[],[]]
euclidianaT = [[],[],[],[]]
rangomin=[[],[],[],[]]
rangomax=[[],[],[],[]]
i=5
tiposNorm = ['',globalequalization,adaptativeequalization,contraststretching]

for col in range(sheet.ncols):
    imgfile = sheet.cell_value(0, col)  
    for col2 in range(sheet.ncols):
        segmenta = sheet.cell_value(1, col)    
        c=c+1
        if c==a:
            segmenta=segmenta
            c=c+1  
            #print(f)
            break
    imgfile=imgfile+'jpg'
    segmenta=segmenta+'jpg'
    for norm in range(len(tiposNorm)):
        img = cv2.imread(imgfile)
        imgNorm = normalizacionMaxMin(img)
        img2 = imgNorm.copy()   
        if norm == 0:
            ima=imgNorm
            ima2=img2
        else:
            ima = tiposNorm[norm](imgNorm)
            ima2 = tiposNorm[norm](img2)
        ima3=img.copy()    
        img1,img2,hista,histb,correlacion,Bhattacharyya, euclidiana=comparacionhistRE(ima,ima2,segmenta)
       
        correlacionT[norm].append(correlacion)
        BhattacharyyaT[norm].append(Bhattacharyya)
        euclidianaT[norm].append(euclidiana)
        rangomin[norm].append(ima3.min())
        rangomax[norm].append(ima3.max())
    
        print(imgfile)
  
    #print('Min: %.3f, Max: %.3f' % (img2.min(), img2.max()))
    #print('Correlación=', correlacion)
    #print('Distancia Bhattacharyya=', Bhattacharyya)

    c=0
