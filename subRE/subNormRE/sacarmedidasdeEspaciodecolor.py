# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:47:42 2019

@author: Nataly
"""
from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from comparacionhistReflejosE import comparacionhistRE

from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio
import cv2
import xlrd
from matplotlib import pyplot as plt
import numpy as np
import pylab as plt 
from matplotlib import pyplot as plt

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

c=0
a=1
r=0
i=0
no=0

correlacionT = np.zeros((166,17))
BhattacharyyaT = np.zeros((166,17))
euclidianaT = np.zeros((166,17))


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
        
        img = read_img(imgfile)
        #img=filtrominimo(img)
        img2 = img.copy()   
        if norm == 0:
             ima = img
             ima2=img2
           
        else:
            ima = tiposNorm[norm](img)
            ima2 = tiposNorm[norm](img2)
        
        if norm==0 or norm>=2:
            print(norm)
            
            ima3=ima2.copy()
            C1,C2,C3=cv2.split(ima3)
            C1a2=C1.copy()
            C2a2=C2.copy()
            C3a2=C3.copy()
            
            img1,img2,hista,histb,correlacion,Bhattacharyya, euclidiana=comparacionhistRE(ima,ima2,segmenta,0)
            correlacionT[i,no]=correlacion
            BhattacharyyaT[i,no]=Bhattacharyya
            euclidianaT[i,no]=euclidiana
            
            img11,img21,hista1,histb1,correlacion1,Bhattacharyya1, euclidiana1=comparacionhistRE(C1,C1a2,segmenta,1)
            correlacionT[i,no+1]=correlacion1
            BhattacharyyaT[i,no+1]=Bhattacharyya1
            euclidianaT[i,no+1]=euclidiana1
            
            img12,img22,hista2,histb2,correlacion2,Bhattacharyya2, euclidiana2=comparacionhistRE(C2,C2a2,segmenta,2)
            correlacionT[i,no+2]=correlacion2
            BhattacharyyaT[i,no+2]=Bhattacharyya2
            euclidianaT[i,no+2]=euclidiana2
            
            img13,img23,hista3,histb3,correlacion3,Bhattacharyya3, euclidiana3=comparacionhistRE(C3,C3a2,segmenta,3)
            correlacionT[i,no+3]=correlacion3
            BhattacharyyaT[i,no+3]=Bhattacharyya3
            euclidianaT[i,no+3]=euclidiana3
            print(i,no,'=',correlacion,',',no+1,'=',correlacion1,', ',no+2,'=',correlacion2,',',no+3,'=',correlacion3)
            if no>=9:
                no=0
            else:
                if norm==0:
                    no=(norm+1)*4
                else:
                    no=(norm)*4
        else:
            img12,img22,hista2,histb2,correlacion2,Bhattacharyya2, euclidiana2=comparacionhistRE(ima,ima2,segmenta,2)
            correlacionT[i,16]=correlacion2
            BhattacharyyaT[i,16]=Bhattacharyya2
            euclidianaT[i,16]=euclidiana2
            
            
    i=i+1
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    c=0
    

#%%
import openpyxl

doc = openpyxl.load_workbook('medidasEspaciosdecolorRE.xlsx')
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
doc.save("medidasEspaciosdecolorRE.xlsx")