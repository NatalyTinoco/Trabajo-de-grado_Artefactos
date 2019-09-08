# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 01:18:58 2019

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

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats

valorintensidad=[]
valorpico=[]
sumas=[]
media=[]
mediana=[]
destan=[]
var=[]
betadm=[]   
asimetria=[]
kurtos=[]
pearson=[]
 
#from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    croppedrgb=im
    croppedrgb_2=croppedrgb.copy()
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    H,S,cropped=cv2.split(HSV)
    
    cropped=Fourier(cropped)
    hist = cv2.calcHist([cropped],[0],None,[256],[0,255])
#    plt.plot(hist)
#    plt.show()
    hisa=hist.copy()
    hist=hist.tolist() 
    u=np.max(hist)
    hi=hist.index(u)
    
    valorintensidad.append(hi)
    valorpico.append(hisa[hi])
   

    if hi==0:
       betadm.append(hisa[hi]/sum(hisa))
    else:
        betadm.append(sum(hisa[0:hi])/sum(hisa))
    asimetria.append(skew(hisa))
    kurtos.append(kurtosis(hisa))
    sumas.append(sum(hisa))
    media.append(np.mean(hisa))
    mediana.append(np.median(hisa))
    destan.append(np.std(hisa))
    var.append(np.var(hisa))



   
import pandas as pd    
datos = {'Valor pico intensidad':valorintensidad,
         'Posición pico intensidad':valorpico,
         'beta':betadm,
         'Asimetria':asimetria,
         'Curtosis':kurtos,
         'sumas':sumas,
         'media':media,
         'mediana':mediana,
         'desviacion E':destan,
         'Varianza':var}
datos = pd.DataFrame(datos)
datos.to_excel('HFOURRE.xlsx') 