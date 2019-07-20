# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:26:23 2019

@author: Nataly
"""


from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from comparacionhistReflejosE import comparacionhistRE
from normalizacióndecontrastelocal import normalizacionlocalcontraste
from Normalizacion import  normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,log
import cv2
import xlrd
from matplotlib import pyplot as plt
import numpy as np

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

c=0
a=1
r=0
i=0
no=0
ran=np.zeros((166,15))

tiposNorm = ['',normalizacionMaxMin,normalizacionrgb,normalizacionlocalcontraste,normalizacionintensi,estandarizacion,log]

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
    
    for norm in range(len(tiposNorm)):
        img = read_img(imgfile)
        if norm == 0:
            ima2=img        
        else:
            ima2 = tiposNorm[norm](img)
        no=(norm*2) +1 
        print('h')
        if norm==0:
            no=0 
       
        
        ran[i,no]=ima2.min()
        ran[i,no+1]=ima2.max()
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

doc = openpyxl.load_workbook('rangosIntensi.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['A','B','C','D','E','F','G','H','I','J','K','L']

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
doc.save("rangosIntensi.xlsx")

