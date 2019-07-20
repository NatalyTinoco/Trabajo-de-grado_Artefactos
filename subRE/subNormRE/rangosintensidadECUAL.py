# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:23:27 2019

@author: Nataly
"""


from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
from pruebaEuclidiana import euclidiana
import cv2
import pylab as plt 
from matplotlib import pyplot as plt
#from scipy import stats # importando scipy.stats
import numpy as np
import glob
from normalizacióndecontrastelocal import normalizacionlocalcontraste
from Normalizacion import  normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,tloga

c=0
a=1
r=0
i=0
no=0
ran=np.zeros((166,15))

tiposNorm = ['',normalizacionMaxMin,normalizacionrgb,normalizacionlocalcontraste,normalizacionintensi,estandarizacion,tloga]

for imgfile in glob.glob("*.jpg"):
    for norm in range(len(tiposNorm)):   
        img= read_img(imgfile)
        if norm == 0:
             ima = img
        else:
            ima = tiposNorm[norm](img)
        no=(norm*2)
        #print('h')
        if norm==0:
            no=0    
        #plt.imshow(ima)
        #plt.show()
        ran[i,no]=ima.min()
        ran[i,no+1]=ima.max()
        print(i,no, ran[i,no])
        print(i,no+1,ran[i,no+1])
        #print(str(tiposNorm[norm]))  
     
        
    i=i+1
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows() 
    c=0
    print(imgfile)
    
#%%

import openpyxl

doc = openpyxl.load_workbook('medidasRangosIntensiEcuaDM.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']

i=0
ii=0
for a in range (int(len(table))):
    #print(i)
    #print(a)
    for x in range ((len(ran[:,1]))):
        hoja[table[i]+ str (x+4)]=ran[x,a]
    #print(table[i])
    #print(table[i+1])
    #print(table[i+2])
    print(x,a)
    i=(a+1)
    #print(a)
doc.save("medidasRangosIntensiEcuaDM.xlsx")
