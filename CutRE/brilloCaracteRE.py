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
from colour import Color

brillomoda = []
brillomedia = []

for image in glob.glob('*.jpg'):
    image = 'bboxreflejo (1).jpg'
    im = cv2.imread(image)
    R,G,B=cv2.split(im)
    width,heigth,ch=im.shape
    ##Brillo
    brillo=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(width*heigth)
    brillomedia.append(np.mean(brillo))
    brillomoda.append(stats.mode(brillo))
    
    #Luminance
    #c1 = Color(R, G, B)
    luminance = (0.2126*im[0]+0.7152*im[1]+0.0722*im[2])
    modeL = stats.mode(luminance)
    meanL = np.mean(luminance)
    ##Varianza
    tamañoW = 5
    tamañoH = 5
    vecesW = int(width/tamañoW)
    vecesH = int(heigth/tamañoH)
    lv = 0
    for w in range(width):
        for h in range(heigth):
            for t1 in range(tamañoW-5):
                cropped = im[t1:t1+tamañoW,t1:t1+tamañoH]
                lv += (im[w+t1,h+t1] - np.mean(cropped))
                print(lv)
#                for t2 in range(tamañoH):
#                    cropped = im[t1:t1+tamañoW,t2:t2+tamañoH]
#                    lv=im[w+t1,h+t2] - np.mean(cropped)
#                    print(lv)

datos = {'Media':brillomedia}

datos = pd.DataFrame(datos)
datos.to_excel('Características_brilloRE.xlsx')
    #plt.imshow(brillo,'Greys')
    #plt.show()
    #print(brillo)