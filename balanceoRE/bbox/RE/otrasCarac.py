# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 23:22:12 2019

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


mediahistH=[]
medianahistH=[]
asimetriahistH=[]
curtosishistH=[]
desviacionhistH=[]
varianzahistH=[]
valorintensidadH=[]
valorpicoH=[]

contrastH=[]
energiH=[]
homogeneiH=[]
correlaciH=[]
disiH=[]
ASH=[]
entropH=[]  



entropiaH=[]
mediaH=[]
medianaH=[]
asimetriaH=[]
curtosisH=[]
l2H=[]
desviacionH=[]
varianzaH=[]
contrasteFraraH=[]
brilloFraraH=[]


mediahistS=[]
medianahistS=[]
asimetriahistS=[]
curtosishistS=[]
desviacionhistS=[]
varianzahistS=[]
valorintensidadS=[]
valorpicoS=[]

contrastS=[]
energiS=[]
homogeneiS=[]
correlaciS=[]
disiS=[]
ASS=[]
entropS=[]  



entropiaS=[]
mediaS=[]
medianaS=[]
asimetriaS=[]
curtosisS=[]
l2S=[]
desviacionS=[]
varianzaS=[]
contrasteFraraS=[]
brilloFraraS=[]

mediahistV=[]
medianahistV=[]
asimetriahistV=[]
curtosishistV=[]
desviacionhistV=[]
varianzahistV=[]
valorintensidadV=[]
valorpicoV=[]

contrastV=[]
energiV=[]
homogeneiV=[]
correlaciV=[]
disiV=[]
ASV=[]
entropV=[]  



entropiaV=[]
mediaV=[]
medianaV=[]
asimetriaV=[]
curtosisV=[]
l2V=[]
desviacionV=[]
varianzaV=[]
contrasteFraraV=[]
brilloFraraV=[]

mediahistB=[]
medianahistB=[]
asimetriahistB=[]
curtosishistB=[]
desviacionhistB=[]
varianzahistB=[]
valorintensidadB=[]
valorpicoB=[]

contrastB=[]
energiB=[]
homogeneiB=[]
correlaciB=[]
disiB=[]
ASB=[]
entropB=[]  


entropiaB=[]
mediaB=[]
medianaB=[]
asimetriaB=[]
curtosisB=[]
l2B=[]
desviacionB=[]
varianzaB=[]
contrasteFraraB=[]
brilloFraraB=[]

for image in glob.glob('*.jpg'):
    #image = 'bboxreflejo (7).jpg'
    #image = '0288-0000583.jpg'
    im = cv2.imread(image)
    imNorm = normalizacionMaxMin(im)
    imEqu = adaptativeequalization(imNorm)
    imeq2=imEqu.copy()
    R,G,B=cv2.split(imEqu)
    H,S,V=cv2.split(imeq2)
    width,heigth,ch=imEqu.shape
    
    """ H"""
   
    
    hist = cv2.calcHist([H],[0],None,[256],[200,255])
    hisa=hist.copy()
    hist=np.asarray(hist).astype(np.int)
    asimetriahistH.append(skew(hisa))
    curtosishistH.append(kurtosis(hisa))
    mediahistH.append(np.mean(hisa))
    medianahistH.append(np.median(hisa))
    desviacionhistH.append(np.std(hisa))
    varianzahistH.append(np.var(hisa))
    l2H.append(LA.norm(hisa))
    histo=hisa.tolist() 
    uHOUR=np.max(histo)
    hi=histo.index(uHOUR)
    valorintensidadH.append(hi)
    valorpicoH.append(hisa[hi])

    g3H,contraste3H,energia3H,homogeneidad3H, correlacion3H, disimi3H, ASM3H,entropia3H=GLCM(H)
    contrastH.append(contraste3H)
    energiH.append(energia3H)
    homogeneiH.append(homogeneidad3H)
    correlaciH.append(correlacion3H)
    disiH.append(disimi3H)
    ASH.append(ASM3H)
    entropH.append(entropia3H)
    
    #brilloHSV=np.sqrt(0.241*H**2+0.691*S**2+0.068*V**2)/(width*heigth)
    entropiaH.append(skimage.measure.shannon_entropy(H)) 
    #brilloH.append(brilloHSV)
    mediaH.append(np.mean(H))
    medianaH.append(np.median(H))
    asimetriaH.append(skew(H))
    curtosisH.append(kurtosis(H))
    desviacionH.append(np.std(H))
    varianzaH.append(np.var(H))
    contrasteFraraH.append(contrastefra(H))
    brilloFraraH.append(brillofra(H))
    
    """ S"""
    histS = cv2.calcHist([S],[0],None,[256],[200,255])
    hisaS=histS.copy()
    histS=np.asarray(histS).astype(np.int)
    asimetriahistS.append(skew(hisaS))
    curtosishistS.append(kurtosis(hisaS))
    mediahistS.append(np.mean(hisaS))
    medianahistS.append(np.median(hisaS))
    desviacionhistS.append(np.std(hisaS))
    varianzahistS.append(np.var(hisaS))
    l2S.append(LA.norm(hisaS))
    histoS=hisaS.tolist() 
    uHOURS=np.max(histoS)
    hiS=histoS.index(uHOURS)
    valorintensidadS.append(hiS)
    valorpicoS.append(hisaS[hiS])

    g3S,contraste3S,energia3S,homogeneidad3S, correlacion3S, disimi3S, ASM3S,entropia3S=GLCM(S)
    contrastS.append(contraste3S)
    energiS.append(energia3S)
    homogeneiS.append(homogeneidad3S)
    correlaciS.append(correlacion3S)
    disiS.append(disimi3S)
    ASS.append(ASM3S)
    entropS.append(entropia3S)
    
    
    entropiaS.append(skimage.measure.shannon_entropy(S)) 
    mediaS.append(np.mean(S))
    medianaS.append(np.median(S))
    asimetriaS.append(skew(S))
    curtosisS.append(kurtosis(S))
    desviacionS.append(np.std(S))
    varianzaS.append(np.var(S))
    contrasteFraraS.append(contrastefra(S))
    brilloFraraS.append(brillofra(S))
    
    """ S"""
    histV = cv2.calcHist([V],[0],None,[256],[200,255])
    hisaV=histV.copy()
    histV=np.asarray(histV).astype(np.int)
    asimetriahistV.append(skew(hisaV))
    curtosishistV.append(kurtosis(hisaV))
    mediahistV.append(np.mean(hisaV))
    medianahistV.append(np.median(hisaV))
    desviacionhistV.append(np.std(hisaV))
    varianzahistV.append(np.var(hisaV))
    l2V.append(LA.norm(hisaV))
    histoV=hisaV.tolist() 
    uHOURV=np.max(histoV)
    hiV=histoV.index(uHOURV)
    valorintensidadV.append(hiV)
    valorpicoV.append(hisaV[hiV])

    g3V,contraste3V,energia3V,homogeneidad3V, correlacion3V, disimi3V, ASM3V,entropia3V=GLCM(V)
    contrastV.append(contraste3V)
    energiV.append(energia3V)
    homogeneiV.append(homogeneidad3V)
    correlaciV.append(correlacion3V)
    disiV.append(disimi3V)
    ASV.append(ASM3V)
    entropV.append(entropia3V)
    
    
    entropiaV.append(skimage.measure.shannon_entropy(V)) 
    mediaV.append(np.mean(V))
    medianaV.append(np.median(V))
    asimetriaV.append(skew(V))
    curtosisV.append(kurtosis(V))
    desviacionV.append(np.std(V))
    varianzaV.append(np.var(V))
    contrasteFraraV.append(contrastefra(V))
    brilloFraraV.append(brillofra(V))
    
    """ S"""
    histB = cv2.calcHist([B],[0],None,[256],[200,255])
    hisaB=histB.copy()
    histB=np.asarray(histB).astype(np.int)
    asimetriahistB.append(skew(hisaB))
    curtosishistB.append(kurtosis(hisaB))
    mediahistB.append(np.mean(hisaB))
    medianahistB.append(np.median(hisaB))
    desviacionhistB.append(np.std(hisaB))
    varianzahistB.append(np.var(hisaB))
    l2B.append(LA.norm(hisaB))
    histoB=hisaB.tolist() 
    uHOURB=np.max(histoB)
    hiB=histoB.index(uHOURB)
    valorintensidadB.append(hiB)
    valorpicoB.append(hisaB[hiB])

    g3B,contraste3B,energia3B,homogeneidad3B, correlacion3B, disimi3B, ASM3B,entropia3B=GLCM(B)
    contrastB.append(contraste3B)
    energiB.append(energia3B)
    homogeneiB.append(homogeneidad3B)
    correlaciB.append(correlacion3B)
    disiB.append(disimi3B)
    ASB.append(ASM3B)
    entropB.append(entropia3B)
    
    
    entropiaB.append(skimage.measure.shannon_entropy(B)) 
    mediaB.append(np.mean(B))
    medianaB.append(np.median(B))
    asimetriaB.append(skew(B))
    curtosisB.append(kurtosis(B))
    desviacionB.append(np.std(B))
    varianzaB.append(np.var(B))
    contrasteFraraB.append(contrastefra(B))
    brilloFraraB.append(brillofra(B))
    
    
    
datos = {'mediahistH':mediahistH,
         'medianahistH':medianahistH,
         'asimetriahistH':asimetriahistH,
         'curtosishistH':curtosishistH,
         'desviacionhistH':desviacionhistH,
         'varianzahistH':varianzahistH,
         'valorintensidadH':valorintensidadH,
         'valorpicoH':valorpicoH,
         'contrastH':contrastH,
         'energiH':energiH,
         'homogeneiH':homogeneiH,
         'correlaciH':correlaciH,
         'disiH':disiH,
         'ASH':ASH,
         'entropH':entropH,
         'entropiaH':entropiaH,
         'mediaH':mediaH,
         'medianaH':medianaH,
         'asimetriaH':asimetriaH,
         'curtosisH':curtosisH,
         'l2H':l2H,
         'desviacionH':desviacionH,
         'varianzaH':varianzaH,
         'contrasteFraraH':contrasteFraraH,
         'brilloFraraH':brilloFraraH,
         
         
         'mediahistS':mediahistS,
         'medianahistS':medianahistS,
         'asimetriahistS':asimetriahistS,
         'curtosishistS':curtosishistS,
         'desviacionhistS':desviacionhistS,
         'varianzahistS':varianzahistS,
         'valorintensidadS':valorintensidadS,
         'valorpicoS':valorpicoS,
         'contrastS':contrastS,
         'energiS':energiS,
         'homogeneiS':homogeneiS,
         'correlaciS':correlaciS,
         'disiS':disiS,
         'ASS':ASS,
         'entropS':entropS,
         'entropiaS':entropiaS,
         'mediaS':mediaS,
         'medianaS':medianaS,
         'asimetriaS':asimetriaS,
         'curtosisS':curtosisS,
         'l2S':l2S,
         'desviacionS':desviacionS,
         'varianzaS':varianzaS,
         'contrasteFraraS':contrasteFraraS,
         'brilloFraraS':brilloFraraS,
         
         'mediahistV':mediahistV,
         'medianahistV':medianahistV,
         'asimetriahistV':asimetriahistV,
         'curtosishistV':curtosishistV,
         'desviacionhistV':desviacionhistV,
         'varianzahistV':varianzahistV,
         'valorintensidadV':valorintensidadV,
         'valorpicoV':valorpicoV,
         'contrastV':contrastV,
         'energiV':energiV,
         'homogeneiV':homogeneiV,
         'correlaciV':correlaciV,
         'disiV':disiV,
         'ASV':ASV,
         'entropV':entropV,
         'entropiaV':entropiaV,
         'mediaV':mediaV,
         'medianaV':medianaV,
         'asimetriaV':asimetriaV,
         'curtosisV':curtosisV,
         'l2V':l2V,
         'desviacionV':desviacionV,
         'varianzaV':varianzaV,
         'contrasteFraraV':contrasteFraraV,
         'brilloFraraV':brilloFraraV,
         
         'mediahistB':mediahistB,
         'medianahistB':medianahistB,
         'asimetriahistB':asimetriahistB,
         'curtosishistB':curtosishistB,
         'desviacionhistB':desviacionhistB,
         'varianzahistB':varianzahistB,
         'valorintensidadB':valorintensidadB,
         'valorpicoB':valorpicoB,
         'contrastB':contrastB,
         'energiB':energiB,
         'homogeneiB':homogeneiB,
         'correlaciB':correlaciB,
         'disiB':disiB,
         'ASB':ASB,
         'entropB':entropB,
         'entropiaB':entropiaB,
         'mediaB':mediaB,
         'medianaB':medianaB,
         'asimetriaB':asimetriaB,
         'curtosisB':curtosisB,
         'l2B':l2B,
         'desviacionB':desviacionB,
         'varianzaB':varianzaB,
         'contrasteFraraB':contrasteFraraB,
         'brilloFraraB':brilloFraraB}
         
datos = pd.DataFrame(datos)
datos.to_excel('Características_OtrasRE.xlsx')
