# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:12:13 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from comparacionhistReflejosE import comparacionhistRE
from normalizacióndecontrastelocal import normalizacionlocalcontraste
from Normalizacion import  normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,log
import cv2
import xlrd
from matplotlib import pyplot as plt

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

c=0
a=1
r=0
i=0

correlacionT = [[],[],[],[],[],[],[]]
BhattacharyyaT = [[],[],[],[],[],[],[]]
euclidianaT = [[],[],[],[],[],[],[]]
rangomin=[[],[],[],[],[],[],[]]
rangomax=[[],[],[],[],[],[],[]]
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
    segmenta=segmenta+'jpg'

    for norm in range(len(tiposNorm)):
        img = read_img(imgfile)
        img2 = img.copy() 
        img3 = img2.copy()
        if norm == 0:
            ima = img
            ima2 = img2
            ima3 = img3
        else:
            ima = tiposNorm[norm](img)
            ima2 = tiposNorm[norm](img2)
            ima3 = tiposNorm[norm](img3)
            
        rangomin[norm].append(ima3.min())
        rangomax[norm].append(ima3.max())
        img1,img2,hista,histb,correlacion,Bhattacharyya, euclidiana=comparacionhistRE(ima,ima2,segmenta)
#        plt.imshow(img1)
#        plt.show
#        plt.imshow(img2)
#        plt.show
        correlacionT[norm].append(correlacion)
        BhattacharyyaT[norm].append(Bhattacharyya)
        euclidianaT[norm].append(euclidiana)
        
        print('Min: %.3f, Max: %.3f' % (ima3.min(), ima3.max()))
        #print('Correlación=', correlacion)
        #print('Distancia Bhattacharyya=', Bhattacharyya)
        #print('Distancia Euclidiana=', euclidiana)
        c=0
        print(imgfile)
    
#%%

import xlsxwriter

workbook = xlsxwriter.Workbook('metricasDistanciaREestesi.xlsx')
worksheet = workbook.add_worksheet('original')

bold = workbook.add_format({'bold': 1})
#table = ['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','']
table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ']

t=0
while t < 20:
    worksheet.write(table[t]+'2', 'Correlación', bold)
    worksheet.write(table[t+1]+'2', 'Bhattacharyya', bold)
    worksheet.write(table[t+2]+'2', 'Euclidiana', bold)
    worksheet.write(table[t+3]+'2', 'Rango Min', bold)
    worksheet.write(table[t+4]+'2', 'Rango Max', bold)
    t += 5
    
row = 2
col=1

for x in range(sheet.ncols):
    worksheet.write(row, col, correlacionT[0][x])
    worksheet.write(row, col+1, BhattacharyyaT[0][x])
    worksheet.write(row, col+2, euclidianaT[0][x]) 
    worksheet.write(row, col+3, rangomin[0][x]) 
    worksheet.write(row, col+4, rangomax[0][x]) 
    
    worksheet.write(row, col+5, correlacionT[1][x])
    worksheet.write(row, col+6, BhattacharyyaT[1][x])
    worksheet.write(row, col+7, euclidianaT[1][x])
    worksheet.write(row, col+8, rangomin[1][x]) 
    worksheet.write(row, col+9, rangomax[1][x]) 
    
    worksheet.write(row, col+10, correlacionT[2][x])
    worksheet.write(row, col+11, BhattacharyyaT[2][x])
    worksheet.write(row, col+12, euclidianaT[2][x])
    worksheet.write(row, col+13, rangomin[2][x]) 
    worksheet.write(row, col+14, rangomax[2][x]) 
    
    
    worksheet.write(row, col+15, correlacionT[3][x])
    worksheet.write(row, col+16, BhattacharyyaT[3][x])
    worksheet.write(row, col+17, euclidianaT[3][x])
    worksheet.write(row, col+18, rangomin[3][x]) 
    worksheet.write(row, col+19, rangomax[3][x]) 
    
    
    worksheet.write(row, col+20, correlacionT[4][x])
    worksheet.write(row, col+21, BhattacharyyaT[4][x])
    worksheet.write(row, col+22, euclidianaT[4][x])
    worksheet.write(row, col+23, rangomin[4][x]) 
    worksheet.write(row, col+24, rangomax[4][x]) 
    
    
    worksheet.write(row, col+25, correlacionT[5][x])
    worksheet.write(row, col+26, BhattacharyyaT[5][x])
    worksheet.write(row, col+27, euclidianaT[5][x])
    worksheet.write(row, col+28, rangomin[5][x]) 
    worksheet.write(row, col+29, rangomax[5][x]) 
    
    
    worksheet.write(row, col+30, correlacionT[6][x])
    worksheet.write(row, col+31, BhattacharyyaT[6][x])
    worksheet.write(row, col+32, euclidianaT[6][x])
    worksheet.write(row, col+33, rangomin[6][x]) 
    worksheet.write(row, col+34, rangomax[6][x]) 
    
    row += 1

workbook.close()
