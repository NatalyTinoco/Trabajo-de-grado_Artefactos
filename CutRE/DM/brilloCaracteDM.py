# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 00:27:48 2019

@author: Usuario
"""

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
import glob 
import pandas as pd
from scipy import stats
from Normalizacion import normalizacionMaxMin
from equalization import adaptativeequalization

brillomoda = []
brillomedia = []
luminanciaMedia = []

for image in glob.glob('*.jpg'):
    #image = '0144-0000603.jpg'
    im = cv2.imread(image)
    imNorm = normalizacionMaxMin(im)
    imEqu = adaptativeequalization(imNorm)
    R,G,B=cv2.split(imEqu)
    width,heigth,ch=imEqu.shape
    
    #Brillo
    brillo=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(w*h)
    brillomedia.append(np.mean(brillo))
    brillomoda.append(stats.mode(brillo))
    
    #Luminancia
    
    luminance = (0.2126*im[0]+0.7152*im[1]+0.0722*im[2])
    modeL = stats.mode(luminance)
    luminanciaMedia.append(np.mean(luminance))

datos = {'Media':brillomedia,
         'Luminacia': luminanciaMedia}

datos = pd.DataFrame(datos)
datos.to_excel('Características_brilloDM.xlsx')
    #plt.imshow(brillo,'Greys')
    #plt.show()
    #print(brillo)