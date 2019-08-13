# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:00:30 2019

@author: Nataly
"""


#from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from comparacionhistReflejosEfiltro import comparacionhistRE

from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio
import cv2
import xlrd
from matplotlib import pyplot as plt

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

c=0
a=1
r=0
i=0

correlacionT = [[],[],[],[],[]]
BhattacharyyaT = [[],[],[],[],[]]
euclidianaT = [[],[],[],[],[]]
i=5
tiposNorm = [filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio]
#tiposNorm = [filtrominimo]
def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    #Adaptative Equalization
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)

    imha=cv2.merge((imhaR,imhaG,imhaB))
    
    return imha
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
        img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        img=adaptativeequalization(img)
        
        img2 = img.copy()   
        ima = tiposNorm[norm](img)
        ima2 = tiposNorm[norm](img2)
        #plt.imshow(img)
        #plt.show()
        img1,img2,hista,histb,correlacion,Bhattacharyya, euclidiana=comparacionhistRE(ima,ima2,segmenta,0)
       
        correlacionT[norm].append(correlacion)
        BhattacharyyaT[norm].append(Bhattacharyya)
        euclidianaT[norm].append(euclidiana)
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
import openpyxl

doc = openpyxl.load_workbook('medidasFiltros.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S']

i=0
ii=0
for a in range (1,int(len(table)/3)+1):
    print(i)
    #print(a)
    for x in range (len(correlacionT[0])):
        hoja[table[i]+ str (x+4)]=correlacionT[ii][x]
        hoja[table[i+1]+ str (x+4)]=BhattacharyyaT[ii][x]
        hoja[table[i+2]+ str (x+4)]=euclidianaT[ii][x]
        print(ii)
    i=a*3
    ii=ii+1
    
doc.save("medidasFiltros.xlsx")

