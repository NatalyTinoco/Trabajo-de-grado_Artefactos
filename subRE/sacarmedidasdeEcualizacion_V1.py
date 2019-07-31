# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:08:32 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
#from pruebaEuclidiana import euclidiana
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
import xlrd

from equalization import globalequalization, adaptativeequalization, contraststretching
from brilloContraste import contraste

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

brillom=np.zeros((166,4))
contrastem=np.zeros((166,4))
i=0
tiposNorm = ['',globalequalization,adaptativeequalization,contraststretching]

for col in range(sheet.ncols):

    imgfile=sheet.cell_value(0, col)
    filetxt=imgfile+'jpg'      
    #bboxfile=filetxt
    #Ecualizaciones
    #img=globalequalization(img)
    #img=adaptativeequalization(img)
    #img=contraststretching(img)
    for norm in range(len(tiposNorm)):
        img = read_img(filetxt)
        #img=filtrominimo(img)
        img2 = img.copy()    
        if norm == 0:
             ima = img
             ima2=img2
           
        else:
            ima = tiposNorm[norm](img)
            ima2 = tiposNorm[norm](img2)
        print(i,norm)   
        ima=cv2.cvtColor(ima, cv2.COLOR_RGB2GRAY) 
        ima2=cv2.cvtColor(ima2, cv2.COLOR_RGB2GRAY) 
        contrastem[i,norm] = contraste(ima)
        #brillom[i,norm]=brillo(ima2)
        #print(imgfile)
        
            
    i=i+1
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    c=0
#%%
import openpyxl

doc = openpyxl.load_workbook('brilloContrasteEcuDM.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
table = ['A','B','C','D','E','F','G','H']
i=0
ii=0
for a in range (int(len(table)/2)):
    #print(i)
    #print(a)
    for x in range ((len(contrastem[:,1]))):
        hoja[table[i]+ str (x+4)]=contrastem[x,a]
        hoja[table[i+1]+ str (x+4)]=brillom[x,a]
    #print(table[i])
    #print(table[i+1])
    #print(table[i+2])
        print(x,a)
    i=(a+1)*2
    #print(a)
doc.save("brilloContrasteEcuDM.xlsx")


"""
   
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
hista = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
img = read_img(imgfile)
img2=img
for z in range((3)):
    img2[:,:,z]=(img2[:,:,z]*(mascara2))  
histb = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
  
imgg=cv2.cvtColor(imgg,cv2.COLOR_RGB2BGR)
img1=cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)

fig, ax = plt.subplots(ncols=3, figsize=(13,7), sharex=True, sharey=True)
ax[0].imshow(imgg)
ax[0].set_title('Log')
ax[1].imshow(img1)
ax[1].set_title('DM')
ax[2].imshow(img2)
ax[2].set_title('Vecindad')
plt.show()

fig, ax = plt.subplots(ncols=3, figsize=(13,7), sharex=True, sharey=True)
ax[0].hist(imgg.ravel(),255,[1,imgg.max()])
ax[1].hist(img1.ravel(),256,[1,img1.max()])
ax[2].hist(img2.ravel(),255,[1,img2.max()])
plt.show()


### Calculo de medidas

#Correlación
correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
medidas[i,0]=correlacion
#Distancia Bhattacharyya
Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA) 
medidas[i,1]=Bhattacharyya
# Euclidiana
eucli=euclidiana(img1,img2)
medidas[i,2]=eucli
#"""