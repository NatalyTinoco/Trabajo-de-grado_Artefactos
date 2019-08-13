# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:07:11 2019

@author: Nataly
"""

#from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from comparacionhistReflejosE import comparacionhistRE

from equalization import globalequalization, adaptativeequalization, contraststretching


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
        img2 = img.copy()   
        if norm == 0:
            ima=img
            ima2=img2
        else:
            ima = tiposNorm[norm](img)
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
    
#%%
#
#import xlsxwriter
#
#workbook = xlsxwriter.Workbook('medidasDistanciEcuRE.xlsx')
#worksheet = workbook.add_worksheet('Hoja1')
#
#bold = workbook.add_format({'bold': 1})
#table = ['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U']
#
#t=0
#while t < 20:
#    worksheet.write(table[t]+'2', 'Correlación', bold)
#    worksheet.write(table[t+1]+'2', 'Bhattacharyya', bold)
#    worksheet.write(table[t+2]+'2', 'Euclidiana', bold)
#    worksheet.write(table[t+3]+'2', 'Rango Min', bold)
#    worksheet.write(table[t+4]+'2', 'Rango Max', bold)
#    t += 5
#    
#row = 2
#col=1
#
#for x in range(sheet.ncols):
#    worksheet.write(row, col, correlacionT[0][x])
#    worksheet.write(row, col+1, BhattacharyyaT[0][x])
#    worksheet.write(row, col+2, euclidianaT[0][x]) 
#    worksheet.write(row, col+3, rangomin[0][x]) 
#    worksheet.write(row, col+4, rangomax[0][x]) 
#    
#    worksheet.write(row, col+5, correlacionT[1][x])
#    worksheet.write(row, col+6, BhattacharyyaT[1][x])
#    worksheet.write(row, col+7, euclidianaT[1][x])
#    worksheet.write(row, col+8, rangomin[1][x]) 
#    worksheet.write(row, col+9, rangomax[1][x]) 
#    
#    worksheet.write(row, col+10, correlacionT[2][x])
#    worksheet.write(row, col+11, BhattacharyyaT[2][x])
#    worksheet.write(row, col+12, euclidianaT[2][x])
#    worksheet.write(row, col+13, rangomin[2][x]) 
#    worksheet.write(row, col+14, rangomax[2][x]) 
#    
#    
#    worksheet.write(row, col+15, correlacionT[3][x])
#    worksheet.write(row, col+16, BhattacharyyaT[3][x])
#    worksheet.write(row, col+17, euclidianaT[3][x])
#    worksheet.write(row, col+18, rangomin[3][x]) 
#    worksheet.write(row, col+19, rangomax[3][x])    
#    row += 1
#
#workbook.close()
#%%
import openpyxl

doc = openpyxl.load_workbook('medidasDistanciEcuRE.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U']

i=0
ii=0
for a in range (int(len(table)/5)):
    #print(i)
    #print(a)
    for x in range (166):
        hoja[table[i]+ str (x+4)]=correlacionT[a][x]
        hoja[table[i+1]+ str (x+4)]=BhattacharyyaT[a][x]
        hoja[table[i+2]+ str (x+4)]=euclidianaT[a][x]
        hoja[table[i+3]+ str (x+4)]=rangomin[a][x]
        hoja[table[i+4]+ str (x+4)]=rangomax[a][x]
    #print(table[i])
    #print(table[i+1])
    #print(table[i+2])
    print(x,a)
    i=(a+1)*5
    #print(a)
doc.save("medidasDistanciEcuRE.xlsx")

