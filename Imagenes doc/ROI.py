# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:04:31 2019

@author: Nataly
"""
import cv2
from matplotlib import pyplot as plt

img=plt.imread('CT56_colitis_06839.jpg')
manual=plt.imread('MANUAL.jpg',0)
automatica=plt.imread('ROI_automatica.jpg',0)
nueva=img.copy()
nuevaA=img.copy()

for z in range(3):
    nueva[:,:,z]=automatica*img[:,:,z]
    nuevaA[:,:,z]=manual[:,:,0]*img[:,:,z]
    
    
    

fig, ax = plt.subplots(ncols=3, figsize=(23,20), sharex=True, sharey=True)
ax[0].imshow(img)

ax[1].imshow(manual,'Greys')
   
ax[2].imshow(nuevaA)

ax[0].set_title('Imagen con etiquetas')
ax[1].set_title('ROI Manual')
ax[2].set_title('Sin etiquetas')

fig, ax = plt.subplots(ncols=3, figsize=(23,20), sharex=True, sharey=True)
ax[0].imshow(img)

ax[1].imshow(manual,'Greys')
   
ax[2].imshow(nueva)

ax[0].set_title('Imagen con etiquetas')
ax[1].set_title('ROI automatica')
ax[2].set_title('Sin etiquetas')


