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
mediahist=[]
medianahist=[]
asimetriahist=[]
curtosishist=[]
puntoderivada=[]
masximaintensidaddes200=[]
l2=[]
desviacionhist=[]
varianzahist=[]
modeL=[]
valorintensidad=[]
valorpico=[]
from numpy.linalg import norm
from scipy.stats import kurtosis
from numpy import linalg as LA
from scipy.stats import skew

def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]

for image in glob.glob('*.jpg'):
    #image = 'bboxreflejo (7).jpg'
    #image = '0288-0000583.jpg'
    im = cv2.imread(image)
    imNorm = normalizacionMaxMin(im)
    imEqu = adaptativeequalization(imNorm)
    R,G,B=cv2.split(imEqu)
    width,heigth,ch=imEqu.shape
    
    brillo=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(width*heigth)
    brillomedia.append(np.mean(brillo))
    brillomoda.append(stats.mode(brillo))
    
    luminance = (0.2126*imEqu[0]+0.7152*imEqu[1]+0.0722*imEqu[2])
    modeL.append(stats.mode(luminance))
    luminanceMedia.append(np.mean(luminance))
    
    entropia.append(skimage.measure.shannon_entropy(imEqu)) 
    
    hist = cv2.calcHist([G],[0],None,[256],[200,255])
    hisa=hist.copy()
    hist=np.asarray(hist).astype(np.int)
    zz=list(range(0,len(hist)))
    for ii in range(len(hist)):
        zz[ii]=int(hist[ii])
    gradiente=np.gradient(zz[:])   
    uu=find_nearest(gradiente,0)
    gradiente=gradiente.tolist()
    umbral1 = gradiente.index(uu)
    umbral=umbral1
    masximaintensidaddes200.append(umbral)
    puntoderivada.append(uu)
    asimetriahist.append(skew(hisa))
    curtosishist.append(kurtosis(hisa))
    mediahist.append(np.mean(hisa))
    medianahist.append(np.median(hisa))
    desviacionhist.append(np.std(hisa))
    varianzahist.append(np.var(hisa))
    l2.append(LA.norm(hisa))
    histo=hisa.tolist() 
    uHOUR=np.max(histo)
    hi=histo.index(uHOUR)
    valorintensidad.append(hi)
    valorpico.append(hisa[hi])
datos = {'Brillo':brillomedia,
         'Luminancia': luminanceMedia,
         'entropia':entropia,
         'Media':mediahist,
         'Mediana':medianahist,
         'Asimetria':asimetriahist,
         'Curtosis':curtosishist,
         'Cambio_Derivada':puntoderivada,
         'Max intensidad pos':masximaintensidaddes200,
         'L2':l2,
         'Desviación':desviacionhist,
         'Varianza':varianzahist,
         'Valor Intensidad max':valorintensidad,
         'Valor pico int':valorpico}

datos = pd.DataFrame(datos)
datos.to_excel('Características_RE.xlsx')
