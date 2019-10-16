# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:57:15 2019

@author: Nataly
"""

import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from brilloContraste import contrastefra, brillofra
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



from numpy.linalg import norm
from scipy.stats import kurtosis
from numpy import linalg as LA
from scipy.stats import skew

def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]
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
brillomedia = []
asimetriahistB=[]
varianzaB=[]
disiB=[]
contrastB=[]
desviacionB=[]
curtosishistB=[]
ASB=[]
entropiaB=[]

for image in glob.glob('*.jpg'):
    im = cv2.imread(image)
    imNorm = normalizacionMaxMin(im)
    imEqu = adaptativeequalization(imNorm)
    R,G,B=cv2.split(imEqu)
    width,heigth,ch=imEqu.shape
    brillo=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(width*heigth)
    brillomedia.append(np.mean(brillo))
    
    histB = cv2.calcHist([B],[0],None,[256],[200,255])
    hisaB=histB.copy()
    histB=np.asarray(histB).astype(np.int)
    asimetriahistB.append(skew(hisaB))
    
    varianzaB.append(np.var(B))
    desviacionB.append(np.std(B))
    curtosishistB.append(kurtosis(hisaB))
    entropiaB.append(skimage.measure.shannon_entropy(B)) 
    
    g3B,contraste3B,energia3B,homogeneidad3B, correlacion3B, disimi3B, ASM3B,entropia3B=GLCM(B)
    contrastB.append(contraste3B)
    disiB.append(disimi3B)
    ASB.append(ASM3B)
datos = {'contrastB':contrastB,
         'desviacionB':desviacionB,         
         'Brillo':brillomedia,
         'curtosishistB':curtosishistB,
         'ASB':ASB,
         'entropiaB':entropiaB,
         'disiB':disiB,
         'varianzaB':varianzaB,
         'asimetriahistB':asimetriahistB}

datos = pd.DataFrame(datos)
datos.to_excel('Características_RE_DM.xlsx')

