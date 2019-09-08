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
    #20
    bb4=median(aa4, disk(10))
    SSIMN.append(ssim(bb4,aa4))
import pandas as pd    
datos = {'Entropia':entrop,
         'sSIMN':SSIMN}   
datos = pd.DataFrame(datos)
datos.to_excel('Caracateristicas_RE_Prueba2.xlsx') 
