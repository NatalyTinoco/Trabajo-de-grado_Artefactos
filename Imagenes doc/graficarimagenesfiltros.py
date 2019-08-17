# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:05:39 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from matplotlib import pyplot as plt
#import glob
from Normalización import normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,tloga
from normalizacióndecontrastelocal import normalizacionlocalcontraste
import cv2
i=0
images =['00000.jpg','0000623.jpg']

for file in images:
    img = "/SIN NADA/" + file
    img = read_img(img)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    imga=img.copy()
    imgb=img.copy()
    imgc=img.copy()
    imgd=img.copy()
    imge=img.copy()
    imgo=img.copy()
    imgg=normalizacionMaxMin(img)
    imga=normalizacionrgb(imga)
    imgb=normalizacionlocalcontraste(imgb)
    imgc=normalizacionintensi(imgc)
    imgd=estandarizacion(imgd)
    imge=tloga(imge) 
    #plt.imshow(imga)
    #plt.show()
    
    fig, ax = plt.subplots(ncols=7, figsize=(13,7), sharex=True, sharey=True)
    ax[0].imshow(imgo)
    
    ax[1].imshow(imgg)
   
    ax[2].imshow(imga)
   
    ax[3].imshow(imgb)
    
    ax[4].imshow(imgc)
    
    ax[5].imshow(imgd)
    
    ax[6].imshow(imge)
    
    if i==0:
        ax[0].set_title('Sin ecualizar')
        ax[1].set_title('Min-Max')
        ax[2].set_title('rgb')
        ax[3].set_title('Local de contraste')    
        ax[4].set_title('Iluminación')
        ax[5].set_title('Estandarización')
        ax[6].set_title('Log')
        
    plt.show()
    i=i+1