# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:54:41 2019

@author: Nataly
"""


from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##

from normalizacióndecontrastelocal import normalizacionlocalcontraste
from Normalización import normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,tloga
from pruebaEuclidiana import euclidiana

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
from scipy import stats # importando scipy.stats
import numpy as np
import glob
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
from scipy.spatial import distance
import statistics

i=0
correlacionT = np.zeros((166,7))
BhattacharyyaT = np.zeros((166,7))
euclidianaT = np.zeros((166,7))
rangomin=np.zeros((166,7))
rangomax=np.zeros((166,7))

tiposNorm = ['',normalizacionMaxMin,normalizacionrgb,normalizacionlocalcontraste,normalizacionintensi,tloga,tloga]


for file in glob.glob("*.jpg"):
    #imgfile=f
    imgfile=file
    filetxt=file[0:len(file)-3]+'txt'      
    bboxfile=filetxt
    for norm in range(len(tiposNorm)):
        img = read_img(imgfile)
        img2 = img.copy() 
        img3 = img2.copy()
        boxes = read_boxes(bboxfile)
        boxes_abs = yolo2voc(boxes, img.shape)  
        if norm == 0:
            img = img
        else:
            img = tiposNorm[norm](img)
        
        ima=img.copy()
        ima1=img.copy()
        pmin, pmax = ima.min (), ima.max ()  
        #print('Min: %.3f, Max: %.3f' % (ima.min(), ima.max()))
            
        ta=img.shape
        ta=list(ta)
        mascara=np.zeros([ta[0],ta[1]])
        mascara2=np.ones([ta[0],ta[1]])
        for b in boxes_abs:
            cls, x1, y1, x2, y2 = b
            if cls == 3:
                artefacto=img[int(y1):int(y2),int(x1):int(x2),:]
                tavta=artefacto.shape
                tavta=list(tavta)
                factor=0                  
                for y in range(int(y1),int(y2)):
                    for x in range(int(x1),int(x2)):
                        mascara[y,x]=1
                        mascara2[y,x]=0
                mascara=mascara.astype(np.uint8)
                mascara2=mascara2.astype(np.uint8)
                img1=img
                
                #plt.imshow(mascara)
                #plt.show()
                #plt.imshow(mascara2)
                #plt.show()
                
                for z in range((3)):
                    img1[:,:,z]=(img1[:,:,z]*(mascara))  
                

                #hista = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],[mini,maxi,mini,maxi,mini,maxi])
                hista = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])
    
                #plt.imshow(img1)
                #plt.show()
               
                img2=ima1                
                for z in range((3)):
                    img2[:,:,z]=(img2[:,:,z]*(mascara2))  

                #histb = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[int(pmin), int(pmax), int(pmin), int(pmax), int(pmin), int(pmax)])                    
                histb = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])
    
                #plt.imshow(img2)
                #plt.show()
               
                ### Calculo de medidas
                #print(i,norm)
                print(file)
                #Correlación
                correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
                correlacionT[i,norm]=correlacion
                #Distancia Bhattacharyya
                Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA) 
                BhattacharyyaT[i,norm]=Bhattacharyya
                # Euclidiana
                eucli=euclidiana(img1,img2)
                euclidianaT[i,norm]=eucli
                rangomin[i,norm]=pmin
                rangomax[i,norm]=pmax
    
    i=i+1
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
#%%
import openpyxl

doc = openpyxl.load_workbook('metricasDistanciaestesi.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI']

i=0
ii=0
for a in range (int(len(table)/5)):
    #print(i)
    #print(a)
    for x in range ((len(correlacionT[:,1]))):
        hoja[table[i]+ str (x+3)]=correlacionT[x,a]
        hoja[table[i+1]+ str (x+3)]=BhattacharyyaT[x,a]
        hoja[table[i+2]+ str (x+3)]=euclidianaT[x,a]
        hoja[table[i+3]+ str (x+3)]=rangomin[x,a]
        hoja[table[i+4]+ str (x+3)]=rangomax[x,a]
    #print(table[i])
    #print(table[i+1])
    #print(table[i+2])
        print(x,a)
    i=(a+1)*5
    #print(a)
doc.save("metricasDistancia.xlsx")