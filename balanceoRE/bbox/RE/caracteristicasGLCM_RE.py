# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:44:06 2019

@author: Nataly
"""
import sys
sys.path.insert(1,'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/funciones')
from equalization import adaptativeequalization

import cv2
import glob 
import numpy as np
import skimage.feature
from skimage.feature import greycoprops
import pandas as pd

#contrastTF = []
#energiTF = []
#homogeneiTF = []
#correlaciTF = []
#disiTF = []
#ASTF = []
#entropTF = []
#imageFile = []

from numpy import linalg as LA
from scipy.stats import skew
from scipy.stats import kurtosis

valorintensidad=[]
valorpico=[]
media=[]
mediana=[]
destan=[]
var=[]
asimetria=[]
kurtos=[]
pendientere=[]
maxderivada=[]
avg_color=[]
l2=[]
from numpy.linalg import norm
from scipy.stats import linregress
import math
#def GLCM (imA):
#    a=int(np.max(imA))
#    g = skimage.feature.greycomatrix(imA, [1], [0], levels=a+1, symmetric=False, normed=True)                  
#    contraste=skimage.feature.greycoprops(g, 'contrast')[0][0]
#    energia=skimage.feature.greycoprops(g, 'energy')[0][0]
#    homogeneidad=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
#    correlacion=skimage.feature.greycoprops(g, 'correlation')[0][0]
#    disimi= greycoprops(g, 'dissimilarity') 
#    ASM= greycoprops(g, 'ASM')
#    entropia=skimage.measure.shannon_entropy(g) 
#    return g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia

for image in glob.glob("*.jpg"):
    im = cv2.imread(image)
    imgNorm = cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    imgEq=adaptativeequalization(imgNorm)
    
    R, G, B = cv2.split(imgEq)
    avg_color_per_row = np.average(imgEq, axis=0) 
    avg_color.append(np.average(avg_color_per_row, axis=0)) 
    hist = cv2.calcHist([G],[0],None,[256],[0,255])
    hisa=hist.copy()
    hist=hist.tolist() 
    uHOUR=np.max(hist)
    hi=hist.index(uHOUR)
    valorintensidad.append(hi)
    valorpico.append(hisa[hi])
    asimetria.append(skew(hisa))
    kurtos.append(kurtosis(hisa))
    media.append(np.mean(hisa))
    mediana.append(np.median(hisa))
    destan.append(np.std(hisa))
    var.append(np.var(hisa))
    gradientec=np.gradient(G)
    alfapma=np.max(gradientec)  
    xsc=np.arange(len(hisa))
    slopec = linregress(xsc, hisa)[0]  # slope in units of y / x
    slope_anglec = math.atan(slopec)  # slope angle in radians
    alfap = math.degrees(slope_anglec) 
    pendientere.append(alfap)
    maxderivada.append(alfapma)
    l2.append(LA.norm(G))
#    print(avg_color)
#    g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(G)
#    contrastTF.append(contraste)
#    energiTF.append(energia)
#    homogeneiTF.append(homogeneidad)
#    correlaciTF.append(correlacion)
#    disiTF.append(disimi)
#    ASTF.append(ASM)
#    entropTF.append(entropia)
#    imageFile.append(image)
#%%
consolidado = pd.DataFrame({"valorintensidad":valorintensidad,
                            "valorpico":valorpico,
                            "media":media
#"Archivo":imageFile,
#                            "contraste": contrastTF,
#                            "Energía": energiTF,
#                            "Homogeneidad":homogeneiTF,
#                            "Correlación": correlaciTF,
#                            "Disimilitud": disiTF,
#                            "ASM": ASTF,
#                            "Entropia":entropTF})


mediana=[]
destan=[]
var=[]
asimetria=[]
kurtos=[]
pendientere=[]
maxderivada=[]
avg_color=[]
l2=[]
    
consolidado.to_excel("caraceterísticas_RE_histograma.xlsx")


