"""
Created on Fri Jul  5 02:05:39 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from matplotlib import pyplot as plt
import numpy as np
import glob
from Normalización import normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,tloga
from normalizacióndecontrastelocal import normalizacionlocalcontraste
import cv2
i=0
images =['WL_00485.jpg']
from PIL import Image

for file in images:
    img = read_img(file)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    imga=img.copy()
#    imgb=img.copy()
#    imgc=img.copy()
#    imgd=img.copy()
    imge=img.copy()
    imgo=img.copy()
#    imgg=normalizacionMaxMin(img)
#    imga=normalizacionrgb(imga)
    print(np.amin(imge),np.amax(imge))
    plt.hist(imgo.ravel(),256,[0,256])
    plt.show()
#    imgb=normalizacionlocalcontraste(imgb)
#    imgc=normalizacionintensi(imgc)
#    imgd=estandarizacion(imgd)
    imge=tloga(imge) 
#    plt.imshow(imga)
#    plt.show()
    
    fig, ax = plt.subplots(ncols=2, figsize=(14,7), sharex=True, sharey=True)
    ax[0].imshow(imgo)
    
#    ax[1].imshow(imgg)
#   
#    ax[1].imshow(imga)
#   
#    ax[3].imshow(imgb)
#    
#    ax[1].imshow(imgc)
#    
#    ax[1].imshow(imgd)
#    
    ax[1].imshow(imge)
    
    if i==0:
        ax[0].set_title('Sin Normalizar')
#        ax[1].set_title('Min-Max')
#        ax[1].set_title('rgb')
#        ax[3].set_title('Local de contraste')    
#        ax[1].set_title('Iluminación')
#        ax[1].set_title('Estandarización')
        ax[1].set_title('Log')
        
    plt.show()
    i=i+1