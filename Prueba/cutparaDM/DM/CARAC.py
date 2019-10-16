# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:58:26 2019

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
def DFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape 
    crow,ccol = int(rows/2) , int(cols/2)
    print(rows,cols,crow,ccol)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back
def Fourier(inA):
    f = np.fft.fft2(inA)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    return fourier 

from scipy.linalg import hadamard
from numpy.linalg import norm
from scipy.stats import linregress
import math
import pylab as py

import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from yolovoc import yolo2voc
from readboxes import read_boxes
from rOI import ROI
from ventaneo import ventaneoo
from agregaceros import agregarceros
from filtromedianaa import filtromediana
import pywt
import pywt.data
from numpy import linalg as LA
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
from skimage.measure import compare_ssim as ssim
from skimage.filters.rank import median
from skimage.morphology import disk


entrop=[]
SSIMN=[]
mediaLH=[]
mediaaltas=[]
asimetriaHDFT=[]
entropiaHH=[]
homogeneiDFT=[]
disi=[]

for imgfile in glob.glob("*.jpg"):
    im = cv2.imread(imgfile)
    aa,bb,c = im.shape    
    croppedrgb=im
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    croppedrgb=im
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    H,cropped,V=cv2.split(HSV)
    cropSinFou=cropped.copy()
    
    g3,contraste3,energia3,homogeneidad3, correlacion3, disimi3, ASM3,entropia3=GLCM(cropSinFou)
    entrop.append(entropia3)
    aa4=cropped.copy()
    bb4=median(aa4, disk(10))
    SSIMN.append(ssim(bb4,aa4))
    
    coeffs2 = pywt.dwt2(cropped, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    bajasfrec=LL.copy()
    altasfrec=LH+HL+HH
    mediaLH.append(np.mean(LH))
    mediaaltas.append(np.mean(altasfrec))
    #dft
    croppedHDFT=DFT(cropped)
    croppedHDFT=croppedHDFT.astype(np.uint8)
    histDFT = cv2.calcHist([croppedHDFT],[0],None,[256],[0,255])
    hisaDFT=histDFT.copy()
    histDFT=histDFT.tolist() 
    uDFT=np.max(histDFT)
    hiDFT=histDFT.index(uDFT)
    asimetriaHDFT.append(skew(hisaDFT))
    
    entropiaHH.append(skimage.measure.shannon_entropy(HH))
    
    cropFou2=DFT(cropped)
    cropFou2= cropFou2.astype(np.uint8)
    g2,contraste2,energia2,homogeneidad2, correlacion2, disimi2, ASM2,entropia2=GLCM(cropFou2)
    homogeneiDFT.append(homogeneidad2)
    
    cropSinFou=cropped.copy()
    g3,contraste3,energia3,homogeneidad3, correlacion3, disimi3, ASM3,entropia3=GLCM(cropSinFou)
    disi.append(disimi3)
    
import pandas as pd    
datos = {'Entropia':entrop,
         'sSIMN':SSIMN,
         'mediaLH':mediaLH,
         'mediaaltas':mediaaltas,
         'asimetriaHDFT':asimetriaHDFT,
         'entropiaHH':entropiaHH,
         'HomogeneidadDFT':homogeneiDFT,
         'DisimilitudSF':disi}  
 
datos = pd.DataFrame(datos)
datos.to_excel('Caracateristicas_DM.xlsx') 
