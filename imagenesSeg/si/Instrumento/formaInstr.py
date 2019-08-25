# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:04:59 2019

@author: Usuario
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

area = []
perimetro = []
centroide = []
convexidad = []
imageBbox = []

for image in glob.glob("*.tif"):
    im = cv2.imread(image,0)
#    im = cv2.resize(im,(500,500))
    im2 = cv2.imread(image)
#    im2 = cv2.resize(im2,(500,500))
    imNorm = cv2.normalize(im,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
    contours,hierarchy = cv2.findContours(imNorm, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in range(len(contours)):
        cnt = contours[c]
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        cv2.drawContours(im2, [approx], -1, (0,255,0), 3)
#        cv2.imshow("Show",im2)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        pp = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
#        plt.imshow(pp)
        M = cv2.moments(approx)
        # centroide
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroide.append((cx,cy))
        else:
            centroide.append((0,0))
        
        convexidad.append(cv2.convexHull(approx))
        area.append(cv2.contourArea(approx))
        perimetro.append(cv2.arcLength(approx,True))
        
        imageBbox.append(image+'-'+str(c))
        
        
        image2 = image.split('.')
        cv2.imwrite('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/imagenesSeg/si/Instrumento/'+image2[0]+'-contorno.jpg',im2)
        
data = {'Archivo':imageBbox,
        'area': area,
        'perimetro': perimetro,
        'centroide': centroide,
        'convexidad': convexidad }   

dataFra = pd.DataFrame(data) 
dataFra.to_excel('caracterForma_inst.xlsx')

#%%
# Print all contours
#    for c in range(len(contours)):
#        areas = cv2.contourArea(contours[c])
#        cnt = contours[c]
#        x,y,w,h = cv2.boundingRect(cnt)
#        epsilon = 0.1*cv2.arcLength(cnt,True)
#        approx = cv2.approxPolyDP(cnt,epsilon,True)
#        cv2.drawContours(im2, [approx], -1, (0,255,0), 3)
##        cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)
#    
#    pp = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
#    plt.imshow(pp)
#    cv2.imshow("Show",im2)
##        
##    
##    cv2.imshow("Show",im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()