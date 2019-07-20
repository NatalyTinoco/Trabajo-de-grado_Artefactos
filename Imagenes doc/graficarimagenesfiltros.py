# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:05:39 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from matplotlib import pyplot as plt
import glob
from equalization import globalequalization, adaptativeequalization, contraststretching
import cv2
i=0
for file in glob.glob("*.jpg"): 
    img = read_img(file)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    imga=img.copy()
    imgc=img.copy()
    imgo=img.copy()
    imgg=globalequalization(img)
    imga=adaptativeequalization(imga)
    imgc=contraststretching(img)
    #plt.imshow(imga)
    #plt.show()
    
    fig, ax = plt.subplots(ncols=4, figsize=(13,7), sharex=True, sharey=True)
    ax[0].imshow(imgo)
    
    ax[1].imshow(imgg)
   
    ax[2].imshow(imga)
   
    ax[3].imshow(imgc)
    
    if i==0:
        ax[0].set_title('Sin ecualizar')
        ax[1].set_title('Global')
        ax[2].set_title('Adaptativa')
        ax[3].set_title('E. Contraste')    
        
    plt.show()
    i=i+1