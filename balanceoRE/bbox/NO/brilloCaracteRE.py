# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 00:27:48 2019

@author: Usuario
"""
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/CutRE')

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
import glob 
import pandas as pd
from scipy import stats
#from colour import Color
from Normalizacion import normalizacionMaxMin
from equalization import adaptativeequalization
from skimage.feature import greycomatrix, greycoprops
import skimage.feature

brillomoda = []
brillomedia = []
luminanceMedia = []
entropia=[]

for image in glob.glob('*.jpg'):
    #image = 'bboxreflejo (7).jpg'
    #image = '0288-0000583.jpg'
    im = cv2.imread(image)
    imNorm = normalizacionMaxMin(im)
    imEqu = adaptativeequalization(imNorm)
    R,G,B=cv2.split(imEqu)
    width,heigth,ch=imEqu.shape
    #plt.hist(imgo.ravel(),256,[0,256])
    #plt.hist(B,256,[0,256])
    #plt.show()
    
    #cv2.imshow('image',B)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    ##Brillo
    brillo=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(width*heigth)
    brillomedia.append(np.mean(brillo))
    brillomoda.append(stats.mode(brillo))
    
    #Luminance
    #c1 = Color(R, G, B)
    luminance = (0.2126*imEqu[0]+0.7152*imEqu[1]+0.0722*imEqu[2])
    modeL = stats.mode(luminance)
    luminanceMedia.append(np.mean(luminance))
    
    entropia.append(skimage.measure.shannon_entropy(imEqu)) 
    ##Varianza
#    tamañoW = 5
#    tamañoH = 5
#    vecesW = int(width/tamañoW)
#    vecesH = int(heigth/tamañoH)
#    lv = 0
#    for w in range(width):
#        for h in range(heigth):
#            for t1 in range(tamañoW-5):
#                cropped = im[t1:t1+tamañoW,t1:t1+tamañoH]
#                lv += (im[w+t1,h+t1] - np.mean(cropped))
#                print(lv)
##                for t2 in range(tamañoH):
##                    cropped = im[t1:t1+tamañoW,t2:t2+tamañoH]
##                    lv=im[w+t1,h+t2] - np.mean(cropped)
##                    print(lv)

datos = {'Brillo':brillomedia,
         'Luminancia': luminanceMedia,
         'entropia':entropia}

datos = pd.DataFrame(datos)
datos.to_excel('Características_brilloNO.xlsx')
    #plt.imshow(brillo,'Greys')
    #plt.show()
    #print(brillo)