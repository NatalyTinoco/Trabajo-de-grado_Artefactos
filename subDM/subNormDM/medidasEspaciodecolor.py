# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 23:26:31 2019

@author: Nataly
"""


from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
from pruebaEuclidiana_ECO import euclidiana
import cv2
import pylab as plt 
from matplotlib import pyplot as plt
#from scipy import stats # importando scipy.stats
import numpy as np
import glob


from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio
from compararDM import compararDM

correlacionT = np.zeros((166,17))
BhattacharyyaT = np.zeros((166,17))
euclidianaT = np.zeros((166,17))
snr=np.zeros((166,16))

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

i=0
def CHSV(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    return img

def CYUV(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    return img

def CXYZ(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2XYZ)
    return img

def CLUV(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
    return img
def GRIS(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return img

tiposNorm = ['',GRIS,CHSV, CYUV, CXYZ]

no=0
for file in glob.glob("*.jpg"):
    for norm in range(len(tiposNorm)):   
        imgfile=file
        filetxt=file[0:len(file)-3]+'txt'      
        bboxfile=filetxt
        img= read_img(imgfile)
        img=filtrominimo(img)
        if norm == 0:
             img = img
             imgC=img.copy()

        else:
            img = tiposNorm[norm](img)
            imgC=img.copy()

        if norm==0 or norm>=2:
            
            ima3=img.copy()
            C1,C2,C3=cv2.split(ima3)
            imgC1=C1.copy()
            imgC2=C2.copy()
            imgC3=C3.copy()
            
            boxes = read_boxes(bboxfile)      
            
            img1,img2,hista,histb=compararDM(img,boxes,imgfile,imgC,0)
            correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
            correlacionT[i,no]=correlacion
            Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA) 
            BhattacharyyaT[i,no]=Bhattacharyya
            eucli=euclidiana(img1,img2,0)
            euclidianaT[i,no]=eucli
            
            boxes = read_boxes(bboxfile) 
            img11,img21,hista1,histb1=compararDM(C1,boxes,imgfile,imgC1,1)
            correlacion1=cv2.compareHist(hista1,histb1,cv2.HISTCMP_CORREL)
            correlacionT[i,no+1]=correlacion1
            Bhattacharyya1=cv2.compareHist(hista1,histb1,cv2.HISTCMP_BHATTACHARYYA) 
            BhattacharyyaT[i,no+1]=Bhattacharyya1
            eucli1=euclidiana(img11,img21,1)
            euclidianaT[i,no+1]=eucli1
            
            boxes = read_boxes(bboxfile) 
            img12,img22,hista2,histb2=compararDM(C2,boxes,imgfile,imgC2,2)
            correlacion2=cv2.compareHist(hista2,histb2,cv2.HISTCMP_CORREL)
            correlacionT[i,no+2]=correlacion2
            Bhattacharyya2=cv2.compareHist(hista2,histb2,cv2.HISTCMP_BHATTACHARYYA) 
            BhattacharyyaT[i,no+2]=Bhattacharyya2
            eucli2=euclidiana(img12,img22,2)
            euclidianaT[i,no+2]=eucli2
            
            
            boxes = read_boxes(bboxfile) 
            img13,img23,hista3,histb3=compararDM(C3,boxes,imgfile,imgC3,3)
            correlacion3=cv2.compareHist(hista3,histb3,cv2.HISTCMP_CORREL)
            correlacionT[i,no+3]=correlacion3
            Bhattacharyya3=cv2.compareHist(hista3,histb3,cv2.HISTCMP_BHATTACHARYYA) 
            BhattacharyyaT[i,no+3]=Bhattacharyya3
            eucli3=euclidiana(img13,img23,3)
            euclidianaT[i,no+3]=eucli3
            print(i,no, no+1,no+2,no+3)
            if no>=9:
                no=0
            else:
                if norm==0:
                    no=(norm+1)*4
                else:
                    no=(norm)*4

            #print(correlacionT[i,no],correlacionT[i,no+1],correlacionT[i,no+2],correlacionT[i,no+3])
            #print(BhattacharyyaT[i,no],BhattacharyyaT[i,no+1],BhattacharyyaT[i,no+2],BhattacharyyaT[i,no+3])
            #print(euclidianaT[i,no],euclidianaT[i,no+1],euclidianaT[i,no+2],euclidianaT[i,no+3])
        else:
            print('gris',norm)
            boxes = read_boxes(bboxfile)      
            
            img1,img2,hista,histb=compararDM(img,boxes,imgfile,imgC,2)
            correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
            correlacionT[i,16]=correlacion
            Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA) 
            BhattacharyyaT[i,16]=Bhattacharyya
            eucli=euclidiana(img1,img2,1)
            euclidianaT[i,16]=eucli
            
        print(imgfile)
        #print(no)

        
            
    i=i+1
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()

#%%
import openpyxl

doc = openpyxl.load_workbook('medidasEspaciosdecolorDM.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY']

i=0
ii=0
for a in range (int(len(table)/3)):
    #print(i)
    #print(a)
    for x in range ((len(correlacionT[:,1]))):
        hoja[table[i]+ str (x+4)]=correlacionT[x,a]
        hoja[table[i+1]+ str (x+4)]=BhattacharyyaT[x,a]
        hoja[table[i+2]+ str (x+4)]=euclidianaT[x,a]
        
        hoja[table[48]+ str (x+4)]=correlacionT[x,16]
        hoja[table[49]+ str (x+4)]=BhattacharyyaT[x,16]
        hoja[table[50]+ str (x+4)]=euclidianaT[x,16]
    #print(table[i])
    #print(table[i+1])
    #print(table[i+2])
    print(x,a)
    i=(a+1)*3
    #print(a)
    
doc.save("medidasEspaciosdecolorDM.xlsx")
#"""