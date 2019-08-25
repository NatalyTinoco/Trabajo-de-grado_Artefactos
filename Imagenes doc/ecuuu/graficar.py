# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:55:33 2019

@author: Nataly
"""
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Imagenes doc/ECU')


from matplotlib import pyplot as plt
import glob
#from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio
from equalization import globalequalization, adaptativeequalization, contraststretching

import cv2
i=0
for file in glob.glob("*.jpg"): 
    img = cv2.imread(file)
    img=cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    imga=img.copy()
    imgc=img.copy()
    imggau=img.copy()
    imgpro=img.copy()
    imgo=img.copy()
    imgg=globalequalization(img)
    imga=adaptativeequalization(imga)
    imgc=contraststretching(imgc)
    #imgg=filtroautolevel(img)
    #imga=filtromediana(imga)
    #imgc=filtrominimo(imgc)
    #imggau=filtrogaussiano(imggau)
    #imgpro=filtropromedio(imgpro)
    plt.imshow(imgg)
    plt.title('Global')
    plt.show()
    plt.hist(imgg.ravel(),256,[0,256])
    plt.show()
    
    plt.imshow(imga)
    plt.title('Adaptativa')
    plt.show()
    plt.hist(imga.ravel(),256,[0,256])
    plt.show()
    
    plt.imshow(imgc)
    plt.title('E. de contrate')
    plt.show()
    plt.hist(imgc.ravel(),256,[0,256])
    plt.show()
    
    
#    fig, ax = plt.subplots(ncols=4, figsize=(16,10), sharex=True, sharey=True)
#    ax[0].imshow(imgo)    
#    ax[1].imshow(imgg)   
#    ax[2].imshow(imga)
#    ax[3].imshow(imgc)
#    #ax[4].imshow(imggau)
#    #ax[5].imshow(imgpro)
#    
#    if i==0:
#        ax[0].set_title('Sin ecualizar')
#        ax[1].set_title('Global')
#        ax[2].set_title('Adaptativa')
#        ax[3].set_title('E. de contrate')
#        #ax[4].set_title('Gaussiano')
#        #ax[5].set_title('Media')
#        
#        
#    plt.show()
#    i=i+1