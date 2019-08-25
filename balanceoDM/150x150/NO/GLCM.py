# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 00:57:49 2019

@author: Nataly
"""


import cv2
import numpy as np
import glob
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from scipy.stats import kurtosis
import statistics as stats
import pywt
import pywt.data
from scipy import ndimage as nd
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel



#tamañoA = []
#tamañoB = []
def Fourier(inA):
    f = np.fft.fft2(inA)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    return fourier 
    
def GLCM (imA):
        a=int(np.max(imA))
        g = skimage.feature.greycomatrix(imA, [1], [0], levels=a+1, symmetric=False, normed=True)                  
        contraste=skimage.feature.greycoprops(g, 'contrast')[0][0]
        energia=skimage.feature.greycoprops(g, 'energy')[0][0]
        homogeneidad=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
        correlacion=skimage.feature.greycoprops(g, 'correlation')[0][0]
        disimi= greycoprops(g, 'dissimilarity') 
        ASM= greycoprops(g, 'ASM')
        entropia=skimage.measure.shannon_entropy(g) 
        return g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia
#                    plt.imshow(cropped)


contrast=[]
energi=[]
homogenei=[]
correlaci=[]
disi=[]
AS=[]
entrop=[]

contrast1=[]
energi1=[]
homogenei1=[]
correlaci1=[]
disi1=[]
AS1=[]
entrop1=[]  
#from matplotlib import pyplot as plt

for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    aa,bb,c = im.shape    
    croppedrgb=im
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    croppedrgb=im
    croppedrgb_2=croppedrgb.copy()
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    H,S,cropped=cv2.split(HSV)
#    H,cropped,V=cv2.split(HSV)
#    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2YUV)
#    H,S,cropped=cv2.split(HSV)
    cropped,U,V=cv2.split(HSV)
    print(cropped.shape)
    cropFou=Fourier(cropped)
    g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(cropFou)
    contrast.append(contraste)
    energi.append(energia)
    homogenei.append(homogeneidad)
    correlaci.append(correlacion)
    disi.append(disimi)
    AS.append(ASM)
    entrop.append(entropia)
    
    g1,contraste1,energia1,homogeneidad1, correlacion1, disimi1, ASM1,entropia1=GLCM(cropped)
    contrast1.append(contraste1)
    energi1.append(energia1)
    homogenei1.append(homogeneidad1)
    correlaci1.append(correlacion1)
    disi1.append(disimi1)
    AS1.append(ASM1)
    entrop1.append(entropia1)
   
import pandas as pd    
datos = {'Contraste':contrast,
         'Energia':energi,
         'Homogeneidad':homogenei,
         'Correlación':correlaci,
         'Disimilitud':disi,
         'ASM':AS,
         'Entropia':entrop,
         'ContrasteSF':contrast1,
         'EnergiaSF':energi1,
         'HomogeneidadSF':homogenei1,
         'CorrelaciónSF':correlaci1,
         'DisimilitudSF':disi1,
         'ASMSF':AS1,
         'EntropiaSF':entrop1}
datos = pd.DataFrame(datos)
#datos.to_excel('GLCMNOS.xlsx') 
datos.to_excel('GLCMNO.xlsx')
#datos.to_excel('GLCMNOY.xlsx')