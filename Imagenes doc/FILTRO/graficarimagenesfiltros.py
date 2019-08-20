# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:05:39 2019

@author: Nataly
"""
def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    #Adaptative Equalization
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)

    imha=cv2.merge((imhaR,imhaG,imhaB))
    
    return imha
from matplotlib import pyplot as plt
from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio

import cv2
i=0
images =['00000.jpg','0000623.jpg']

for file in images:
    img =  file
    img = cv2.imread(img)
    img=cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    if file=='00000.jpg':
        img=adaptativeequalization(img)
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        plt.imshow(img)
        plt.show()
        
    else:
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        img=img
    
    imga=img.copy()
    imgb=img.copy()
    imgc=img.copy()
    imgd=img.copy()
    imge=img.copy()
    imgo=img.copy()
    imgg=filtroautolevel(img)
    imga=filtromediana(imga)
    imgb=filtrominimo(imgb)
    imgc=filtrogaussiano(imgc)
    imgd=filtropromedio(imgd)

    
    fig, ax = plt.subplots(ncols=6, figsize=(20,13), sharex=True, sharey=True)
    ax[0].imshow(imgo)
    
    ax[1].imshow(imgg)
   
    ax[2].imshow(imga)
   
    ax[3].imshow(imgb)
    
    ax[4].imshow(imgc)
    
    ax[5].imshow(imgd)
    

    if i==0:
        ax[0].set_title('Sin filtrar')
        ax[1].set_title('Auto nivel')
        ax[2].set_title('Mediana')
        ax[3].set_title('Mínimo')    
        ax[4].set_title('Gaussiano')
        ax[5].set_title('Media')
        
    plt.show()
    i=i+1