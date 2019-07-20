# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:08:32 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
from plotboxes import plot_boxes #graficar bbox ## plot_boxes(ax, boxes)##
from readobjnames import read_obj_names  ## classes = read_obj_names(classfile) ##
from normalizacióndecontrastelocal import normalizacionlocalcontraste
from Normalización import normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,tloga
from pruebaEuclidiana import euclidiana
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

medidas=np.zeros((166,3))
i=0

for file in glob.glob("*.jpg"):
    #imgfile=f
    imgfile=file
    filetxt=file[0:len(file)-3]+'txt'      
    bboxfile=filetxt
    img = read_img(imgfile)
    #Normalizaciones
    #img=normalizacionrgb(img)
    #img=normalizacionlocalcontraste(img)
    #img=normalizacionintensi(img)
    #img=estandarizacion(img).astype(np.uint8)
    #img=tloga(img)
    #img=normalizacionMaxMin(img)
    #pmin, pmax = img.min (), img.max ()
    #img =  256 * (img - pmin) / (pmax - pmin)
    #img=img.astype(np.uint8)
    imgg=img.copy()
    #print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
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
                    #print(y,x)
                    mascara[y,x]=1
                    mascara2[y,x]=0
            # mascara=mascara[int(y1-factor):int(y2+factor),int(x1-factor):int(x2+factor)]
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
    #"""
   
    imgg=cv2.cvtColor(imgg,cv2.COLOR_RGB2BGR)
    img1=cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
    
    fig, ax = plt.subplots(ncols=3, figsize=(13,7), sharex=True, sharey=True)
    ax[0].imshow(imgg)
    ax[0].set_title('RGB')
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
    
    #"""
    ### Calculo de medidas
    
    #Correlación
    correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
    """ 
    histgramaNor = []
    histograma2Nor = []
    euclidianNor = []


    for d in range(3):
        histNor = cv2.calcHist([img1], [d], None, [256], [1, 256])
        hist2Nor = cv2.calcHist([img2], [d], None, [256], [1, 256])
        histgramaNor.append(histNor)
        histograma2Nor.append(hist2Nor) 
    
    for n in range(3):
        euclNor = cv2.compareHist(histgramaNor[n], histograma2Nor[n],cv2.HISTCMP_CORREL)
        euclidianNor.append(euclNor)  

        
    correlacion=statistics.mean(euclidianNor)
    """
    medidas[i,0]=correlacion
    #Distancia Bhattacharyya
    Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA) 
    medidas[i,1]=Bhattacharyya
    # Euclidiana
    eucli=euclidiana(img1,img2)
    medidas[i,2]=eucli
    
    i=i+1
    #print(file)
    print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
#%%
#"""
import openpyxl
doc = openpyxl.load_workbook('metricasDistancia.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
for x in range (len(medidas)):
    hoja['W'+ str (x+4)]=medidas[x,0]
    hoja['X'+ str (x+4)]=medidas[x,1]
    hoja['Y'+ str (x+4)]=medidas[x,2]

doc.save("metricasDistancia.xlsx")

#"""
