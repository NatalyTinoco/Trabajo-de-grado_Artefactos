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
tiposNorm = ['',normalizacionMaxMin,normalizacionintensi,normalizacionlocalcontraste,estandarizacion,normalizacionrgb]

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
        if norm == 0:
            ima = img
            ima2 = img2
        else:
            ima = tiposNorm[norm](img)
            ima2 = tiposNorm[norm](img2)
        img1,img2,hista,histb,correlacion,Bhattacharyya, euclidiana=comparacionhistRE(ima,ima2,segmenta)
        correlacionT[norm].append(correlacion)
        BhattacharyyaT[norm].append(Bhattacharyya)
        euclidianaT[norm].append(euclidiana)
        
        #print('Min: %.3f, Max: %.3f' % (img2.min(), img2.max()))
        #print('Correlación=', correlacion)
        #print('Distancia Bhattacharyya=', Bhattacharyya)
        #print('Distancia Euclidiana=', euclidiana)
        c=0
        print(imgfile)
    
#%%

import xlsxwriter

workbook = xlsxwriter.Workbook('test.xlsx')
worksheet = workbook.add_worksheet('original')

bold = workbook.add_format({'bold': 1})
table = ['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V']
t=0
while t < 20:
    worksheet.write(table[t]+'2', 'Correlación', bold)
    worksheet.write(table[t+1]+'2', 'Bhattacharyya', bold)
    worksheet.write(table[t+2]+'2', 'Euclidiana', bold)
    t += 3
    
row = 2
col=1

for x in range(sheet.ncols):
    worksheet.write(row, col, correlacionT[0][x])
    worksheet.write(row, col+1, BhattacharyyaT[0][x])
    worksheet.write(row, col+2, euclidianaT[0][x]) 
    worksheet.write(row, col+3, correlacionT[1][x])
    worksheet.write(row, col+4, BhattacharyyaT[1][x])
    worksheet.write(row, col+5, euclidianaT[1][x])
    worksheet.write(row, col+6, correlacionT[2][x])
    worksheet.write(row, col+7, BhattacharyyaT[2][x])
    worksheet.write(row, col+8, euclidianaT[2][x])
    worksheet.write(row, col+9, correlacionT[3][x])
    worksheet.write(row, col+10, BhattacharyyaT[3][x])
    worksheet.write(row, col+11, euclidianaT[3][x])
    worksheet.write(row, col+12, correlacionT[4][x])
    worksheet.write(row, col+13, BhattacharyyaT[4][x])
    worksheet.write(row, col+14, euclidianaT[4][x])
    worksheet.write(row, col+15, correlacionT[5][x])
    worksheet.write(row, col+16, BhattacharyyaT[5][x])
    worksheet.write(row, col+17, euclidianaT[5][x])
    worksheet.write(row, col+18, correlacionT[6][x])
    worksheet.write(row, col+19, BhattacharyyaT[6][x])
    worksheet.write(row, col+20, euclidianaT[6][x])
    row += 1

workbook.close()
