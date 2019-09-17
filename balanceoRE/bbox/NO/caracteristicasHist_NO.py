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
    #xsc=np.arange(len(hisa))
    #slopec = linregress(xsc, hisa)[0]  # slope in units of y / x
    #slope_anglec = math.atan(slopec)  # slope angle in radians
    #alfap = math.degrees(slope_anglec) 
    #pendientere.append(alfap)
    maxderivada.append(alfapma)
    l2.append(LA.norm(G))

#%%
consolidado = pd.DataFrame({"valorintensidad":valorintensidad,
                            "valorpico":valorpico,
                            "media":media,
                            "mediana":mediana,
                            "destan":destan,
                            "var":var,
                            "asimetria":asimetria,
                            "kurtos":kurtos,
                            "maxderivada":maxderivada,
                            "avg_color":avg_color,
                            "l2":l2})
    
consolidado.to_excel("caraceterísticas_NO_histograma.xlsx")


