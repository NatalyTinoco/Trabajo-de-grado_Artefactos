# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:00:30 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
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

from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio


correlacionT = np.zeros((166,5))
BhattacharyyaT = np.zeros((166,5))
euclidianaT = np.zeros((166,5))

i=0
tiposNorm = [filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio]


for file in glob.glob("*.jpg"):
    for norm in range(len(tiposNorm)):        
            imgfile=file
            filetxt=file[0:len(file)-3]+'txt'      
            bboxfile=filetxt
            ima = read_img(imgfile)
            img=ima.copy()
            img=tiposNorm[norm](img)
            boxes = read_boxes(bboxfile)
            boxes_abs = yolo2voc(boxes, img.shape)  
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
            for z in range((3)):
                img1[:,:,z]=(img1[:,:,z]*(mascara))  
            hista = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])
            img = read_img(imgfile)
            img2=img
            for z in range((3)):
                img2[:,:,z]=(img2[:,:,z]*(mascara2))  
            histb = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])   
            ### Calculo de medidas
            #bordes = cv2.Canny(img,180,260)
#            plt.imshow(bordes)
#            plt.show()
            #Correlación
            correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
            correlacionT[i,norm]=correlacion
            #Distancia Bhattacharyya
            Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA) 
            BhattacharyyaT[i,norm]=Bhattacharyya
            # Euclidiana
            eucli=euclidiana(img1,img2)
            euclidianaT[i,norm]=eucli
            #print(i,norm)
            print(imgfile)
            #print(tiposNorm[norm])
    i=i+1
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()

#%%
import openpyxl

doc = openpyxl.load_workbook('medidasFiltrosDMestesi.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S']

i=0
ii=0
for a in range (1,int(len(table)/3)+1):
    print(i)
    #print(a)
    for x in range ((len(correlacionT[:,1]))):
        hoja[table[i]+ str (x+4)]=correlacionT[x,ii]
        hoja[table[i+1]+ str (x+4)]=BhattacharyyaT[x,ii]
        hoja[table[i+2]+ str (x+4)]=euclidianaT[x,ii]
    i=a*3
    ii=ii+1
doc.save("medidasFiltrosDM.xlsx")



