# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:49:09 2019

@author: Nataly
"""
#from pathlib import Path
#datafolder=Path("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/txt")
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from Filtros import filtromediana
# filtromediana, filtrominimo,filtrogaussiano,filtropromedio,
import cv2 
#import matplotlib.pyplot as plt
file='00000'
img=cv2.imread(file+'.jpg')
img=cv2.resize(img,(600,620))
imgoriginal=img.copy()
mask =cv2.imread(file+'_seg.jpg',0)
mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
mask=cv2.resize(mask,(600,620))
nu=20
img_suav= filtromediana(img,nu)


#cv2.imshow('mascara',mask)
#cv2.waitKey(0)

cv2.imshow('suavizada',img_suav)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask_suav=img.copy()
for z in range(3):
    mask_suav[:,:,z]=img_suav[:,:,z]*mask
    


cv2.imshow('mascara',mask_suav)
cv2.waitKey(0)
cv2.destroyAllWindows()

mR,mG,mB=cv2.split(mask_suav)
sum_mask=mR+mG+mB

imagen_2=img.copy()
for z in range(3):
    imagen_2[:,:,z]=imgoriginal[:,:,z]*(1-mask)

cv2.imshow('mascara', imagen_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_result=imagen_2+mask_suav  
cv2.imshow('mascara',img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

dire='./'+file+'_suavizado_'+str(nu)+'.jpg'
cv2.imwrite(dire,img_result)
k = cv2.waitKey(1000)
cv2.destroyAllWindows()
