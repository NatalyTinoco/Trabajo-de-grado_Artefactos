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

contrastTF = []
energiTF = []
homogeneiTF = []
correlaciTF = []
disiTF = []
ASTF = []
entropTF = []
imageFile = []

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

for image in glob.glob("*.jpg"):
    im = cv2.imread(image)
    imgNorm = cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    imgEq=adaptativeequalization(imgNorm)
    
    R, G, B = cv2.split(imgEq)
    
    g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(G)
    contrastTF.append(contraste)
    energiTF.append(energia)
    homogeneiTF.append(homogeneidad)
    correlaciTF.append(correlacion)
    disiTF.append(disimi)
    ASTF.append(ASM)
    entropTF.append(entropia)
    imageFile.append(image)
#%%
consolidado = pd.DataFrame({"Archivo":imageFile,
                            "contraste": contrastTF,
                            "Energía": energiTF,
                            "Homogeneidad":homogeneiTF,
                            "Correlación": correlaciTF,
                            "Disimilitud": disiTF,
                            "ASM": ASTF,
                            "Entropia":entropTF})
    
consolidado.to_excel("caraceterísticas_NO_GLCM.xlsx")


