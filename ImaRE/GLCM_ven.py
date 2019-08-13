# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:27:26 2019

@author: Nataly
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from skimage import data
def tloga(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img

def normalizacionMaxMin(img):
    img2 = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img2

import glob

contraste=[]
energia=[]
homogeneidad=[]
correlacion=[]
disimi=[]
ASM=[]
entropia=[]
i=0               
for file in glob.glob("*.jpg"): 
    img=cv2.imread(file)
    img=normalizacionMaxMin(img)
    #plt.imshow(img) 
    #plt.show()
    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(HSV)
#    f = np.fft.fft2(V)
#    fshift = np.fft.fftshift(f)
#    fourier = 20*np.log(np.abs(fshift))
#    fourier=fourier.astype(np.uint8)
    fourier=V
    #plt.imshow(fourier, cmap = 'gray')
    #plt.show()
    a=int(np.max(fourier))
    g = skimage.feature.greycomatrix(fourier, [1], [0], levels=a+1, symmetric=False, normed=True) 
   
    contraste.append(skimage.feature.greycoprops(g, 'contrast')[0][0])
    energia.append(skimage.feature.greycoprops(g, 'energy')[0][0])
    homogeneidad.append(skimage.feature.greycoprops(g, 'homogeneity')[0][0])
    correlacion.append(skimage.feature.greycoprops(g, 'correlation')[0][0])
    disimi.append(greycoprops(g, 'dissimilarity') )
    ASM.append(greycoprops(g, 'ASM'))
    entropia.append(skimage.measure.shannon_entropy(g))
    i=i+1
import pandas as pd    
datos = {'Contraste':contraste,
         #'Brillo Opción 2':brilloop2,
         'Energia':energia,
         'Homogeneidad':homogeneidad,
         'Correlación':correlacion,
         'Disimilitud':disimi,
         'ASM':ASM,
         'Entropía':entropia}

datos = pd.DataFrame(datos)
datos.to_excel('GlCM_HSV_RETODA_Sinfourier.xlsx')


        
        
        
        