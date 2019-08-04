# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:12:13 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from comparacionhistReflejosE import comparacionhistRE

from equalization import globalequalization, adaptativeequalization, contraststretching
from contraste_V0 import contraste

import cv2
import xlrd
from matplotlib import pyplot as plt

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

c=0
a=1
r=0
i=0

correlacionT = [[],[],[]]
BhattacharyyaT = [[],[],[]]
euclidianaT = [[],[],[]]
contrast = [[],[],[]]
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
    img = read_img(imgfile)
    img2 = img.copy()    
    for norm in range(len(tiposNorm)):
        if norm == 0: 
            ima = img
            ima2 = img2
        else:
            ima = tiposNorm[norm](img)
            ima2 = tiposNorm[norm](img2)
        
        ima=cv2.cvtColor(ima, cv2.COLOR_RGB2GRAY)
        contr = contraste(ima)
        img1,img2,hista,histb,correlacion,Bhattacharyya, euclidiana=comparacionhistRE(ima,ima2,segmenta)
       
        correlacionT[norm].append(correlacion)
        BhattacharyyaT[norm].append(Bhattacharyya)
        euclidianaT[norm].append(euclidiana)
        contrast[norm].append(contr)
        print(norm)
    """
    imgg=cv2.cvtColor(imgg,cv2.COLOR_RGB2BGR)
    img1=cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
    fig, ax = plt.subplots(ncols=3, figsize=(13,7), sharex=True, sharey=True)
    ax[0].imshow(imgg)
    ax[0].set_title('Normalización local de contraste')
    ax[1].imshow(img1)
    ax[1].set_title('RE')
    ax[2].imshow(img2)
    ax[2].set_title('Vecindad')
    plt.show()
    
    fig, ax = plt.subplots(ncols=3, figsize=(13,7), sharex=True, sharey=True)
    ax[0].hist(imgg.ravel(),255,[imgg.min(),imgg.max()])
    ax[1].hist(img1.ravel(),256,[img1.min(),img1.max()])
    ax[2].hist(img2.ravel(),255,[img2.min(),img2.max()])
    plt.show()
    
    #print('Min: %.3f, Max: %.3f' % (img2.min(), img2.max()))
    #print('Correlación=', correlacion)
    #print('Distancia Bhattacharyya=', Bhattacharyya)
    #print('Distancia Euclidiana=', euclidiana)
    """
    c=0
    print(imgfile)
    
#%%

import xlsxwriter

workbook = xlsxwriter.Workbook('test1.xlsx')
worksheet = workbook.add_worksheet('Hoja1')

bold = workbook.add_format({'bold': 1})
table = ['L','M','N','O','P','Q','R','S','T']
t=0
while t < 8:
    worksheet.write(table[t]+'2', 'Correlación', bold)
    worksheet.write(table[t+1]+'2', 'Bhattacharyya', bold)
    worksheet.write(table[t+2]+'2', 'Euclidiana', bold)
    t += 3
    
row = 2
col=1
#%%
for x in range(sheet.ncols):
    worksheet.write(row, col+10, correlacionT[0][x])
    worksheet.write(row, col+11, BhattacharyyaT[0][x])
    worksheet.write(row, col+12, euclidianaT[0][x]) 
    worksheet.write(row, col+13, correlacionT[1][x])
    worksheet.write(row, col+14, BhattacharyyaT[1][x])
    worksheet.write(row, col+15, euclidianaT[1][x])
    worksheet.write(row, col+16, correlacionT[2][x])
    worksheet.write(row, col+17, BhattacharyyaT[2][x])
    worksheet.write(row, col+18, euclidianaT[2][x])
    row += 1

workbook.close()

