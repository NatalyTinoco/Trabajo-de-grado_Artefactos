# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:07:47 2019

@author: Nataly
"""

#from readimg import read_img
import cv2
import numpy as np
import glob
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from scipy.stats import kurtosis
import statistics as stats
import pywt
import pywt.data

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

energiaDM_LH=[]
homogeneidadDM_LH=[]
correlacionDM_LH=[]
disimiDM_LH=[]
ASMDM_LH=[]
entropiaDM_LH=[]
contrasteDM_LH=[]
mediaglcmDM_LH=[]   
entropianoglcmDM_LH=[]
mediaDM_LH=[]
modaDM_LH=[]    
desviacionDM_LH=[]
curtosisDM_LH=[]
nergianoglcmDM_LH=[]
 
 ##GLCMsinfourierLL
energiaDM_SF_LH=[]
homogeneidadDM_SF_LH=[]
correlacionDM_SF_LH=[]
disimiDM_SF_LH=[]
ASMDM_SF_LH=[]
entropiaDM_SF_LH=[]
contrasteDM_SF_LH=[]
mediaglcmDM_SF_LH=[]
entropianoglcmDM_SF_LH=[]
mediaDM_SF_LH=[]
modaDM_SF_LH=[]
desviacionDM_SF_LH=[]
curtosisDM_SF_LH=[]
nergianoglcmDM_SF_LH=[]

 ## HLrojoo
energiaDM_HLROJO=[]
homogeneidadDM_HLROJO=[]
correlacionDM_HLROJO=[]
disimiDM_HLROJO=[]
ASMDM_HLROJO=[]
entropiaDM_HLROJO=[]
contrasteDM_HLROJO=[]
mediaglcmDM_HLROJO=[]
entropianoglcmDM_HLROJO=[]
mediaDM_HLROJO=[]
modaDM_HLROJO=[]
desviacionDM_HLROJO=[]
curtosisDM_HLROJO=[]
nergianoglcmDM_HLROJO=[]
 
##GLCMsinfourierLL
energiaDM_SF_HLROJO=[]
homogeneidadDM_SF_HLROJO=[]
correlacionDM_SF_HLROJO=[]
disimiDM_SF_HLROJO=[]
ASMDM_SF_HLROJO=[]
entropiaDM_SF_HLROJO=[]
contrasteDM_SF_HLROJO=[]
mediaglcmDM_SF_HLROJO=[]
entropianoglcmDM_SF_HLROJO=[]
mediaDM_SF_HLROJO=[]
modaDM_SF_HLROJO=[]
desviacionDM_SF_HLROJO=[]
curtosisDM_SF_HLROJO=[]
nergianoglcmDM_SF_HLROJO=[]         
        
  
from matplotlib import pyplot as plt

for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    aa,bb,c = im.shape    
    croppedrgb=im
    #V=cropped
    cropped=cv2.imread('./V/'+image,0)
    """ #Con Fourier No rojo"""
    coeffs2 = pywt.dwt2(cropped, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    LH=LH.astype(np.uint8)
    cropFou=Fourier(LH)
    ch= cropFou.shape
   
    if len(ch)>2:
        cropFou=cropFou[:,:,0]
    print(cropFou.shape)  
#    plt.imshow(LH,'Greys')
#    plt.show()
    cropped_1=cropped.copy()
    croppedrgb_1=croppedrgb.copy()
    
    g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(cropFou)
    contrasteDM_LH.append(contraste)
    energiaDM_LH.append(energia)
    print(energia)
    homogeneidadDM_LH.append(homogeneidad)
    correlacionDM_LH.append(correlacion)
    disimiDM_LH.append(disimi)
    ASMDM_LH.append(ASM)
    entropiaDM_LH.append(entropia)
    print(entropia)
    mediaglcmDM_LH.append(np.mean(g))   
    entropianoglcmDM_LH.append(skimage.measure.shannon_entropy(cropFou))
    mediaDM_LH.append(np.mean(cropFou))    
    desviacionDM_LH.append(np.var(cropFou))
    modaDM_LH.append(np.median(g))
    curtosisDM_LH.append(np.cov(cropFou))
    nergianoglcmDM_LH.append(np.median(cropFou))
    """ #Sin Fourier No rojo"""
    coeffs2_1 = pywt.dwt2(cropped_1, 'bior1.3')
    LL_1, (LH_1, HL_1, HH_1) = coeffs2_1
    LH_1=LH_1.astype(np.uint8)
    cropFou_1=LH_1
    ch= cropFou_1.shape
    if len(ch)>2:
        cropFou_1=cropFou_1[:,:,0]
    
    g_1,contraste_1,energia_1,homogeneidad_1, correlacion_1, disimi_1, ASM_1,entropia_1=GLCM(cropFou_1)
    contrasteDM_SF_LH.append(contraste_1)
    energiaDM_SF_LH.append(energia_1)
    homogeneidadDM_SF_LH.append(homogeneidad_1)
    correlacionDM_SF_LH.append(correlacion_1)
    disimiDM_SF_LH.append(disimi_1)
    ASMDM_SF_LH.append(ASM_1)
    entropiaDM_SF_LH.append(entropia_1)
    mediaglcmDM_SF_LH.append(np.mean(g_1))   
    entropianoglcmDM_SF_LH.append(skimage.measure.shannon_entropy(cropFou_1))
    mediaDM_SF_LH.append(np.mean(cropFou_1))
    #modaDM_SF_LH.append(np.mean(stats.mode(cropFou_1)))    
    desviacionDM_SF_LH.append(np.var(cropFou_1))
    #curtosisDM_SF_LH.append(kurtosis(cropFou_1))
    modaDM_SF_LH.append(np.median(g_1))
    curtosisDM_SF_LH.append(np.cov(cropFou_1))
    nergianoglcmDM_SF_LH.append(np.median(cropFou_1))
    ###ROJO
    """ #Con Fourier  rojo"""
    coeffs2rgb = pywt.dwt2(croppedrgb, 'bior1.3')
    LLrgb, (LHrgb, HLrgb, HHrgb) = coeffs2rgb
    HLrgb=HLrgb.astype(np.uint8)
    cropFourgb=Fourier(HLrgb)
    ch= cropFourgb.shape
    if len(ch)>2:
        cropFourgb=cropFourgb[:,:,0]
    
    grgb,contrastergb,energiargb,homogeneidadrgb, correlacionrgb, disimirgb, ASMrgb,entropiargb=GLCM(cropFourgb)
    contrasteDM_HLROJO.append(contrastergb)
    energiaDM_HLROJO.append(energiargb)
    homogeneidadDM_HLROJO.append(homogeneidadrgb)
    correlacionDM_HLROJO.append(correlacionrgb)
    disimiDM_HLROJO.append(disimirgb)
    ASMDM_HLROJO.append(ASMrgb)
    entropiaDM_HLROJO.append(entropiargb)
    mediaglcmDM_HLROJO.append(np.mean(grgb))   
    entropianoglcmDM_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb))
    mediaDM_HLROJO.append(np.mean(cropFourgb))
    #modaDM_HLROJO.append(np.mean(stats.mode(cropFourgb)))    
    desviacionDM_HLROJO.append(np.var(cropFourgb))
    #curtosisDM_HLROJO.append(kurtosis(cropFourgb))
    modaDM_HLROJO.append(np.median(grgb))
    curtosisDM_HLROJO.append(np.cov(cropFourgb))
    nergianoglcmDM_HLROJO.append(np.median(cropFourgb))
    """ #Sin Fourier rojo"""
    coeffs2rgb_1 = pywt.dwt2(croppedrgb_1, 'bior1.3')
    LLrgb_1, (LHrgb_1, HLrgb_1, HHrgb_1) = coeffs2rgb_1
    HLrgb_1=HLrgb_1.astype(np.uint8)
    cropFourgb_1=HLrgb_1
    ch=cropFourgb_1.shape
    if len(ch)>2:
        cropFourgb_1=cropFourgb_1[:,:,0]
    
    grgb_1,contrastergb_1,energiargb_1,homogeneidadrgb_1, correlacionrgb_1, disimirgb_1, ASMrgb_1,entropiargb_1=GLCM(cropFourgb_1)
    contrasteDM_SF_HLROJO.append(contrastergb_1)
    energiaDM_SF_HLROJO.append(energiargb_1)
    homogeneidadDM_SF_HLROJO.append(homogeneidadrgb_1)
    correlacionDM_SF_HLROJO.append(correlacionrgb_1)
    disimiDM_SF_HLROJO.append(disimirgb_1)
    ASMDM_SF_HLROJO.append(ASMrgb_1)
    entropiaDM_SF_HLROJO.append(entropiargb_1)
    mediaglcmDM_SF_HLROJO.append(np.mean(grgb_1))   
    entropianoglcmDM_SF_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb_1))
    mediaDM_SF_HLROJO.append(np.mean(cropFourgb_1))
    #modaDM_SF_HLROJO.append(np.mean(stats.mode(cropFourgb_1)))    
    desviacionDM_SF_HLROJO.append(np.var(cropFourgb_1))
    #curtosisDM_SF_HLROJO.append(kurtosis(cropFourgb_1))
    modaDM_SF_HLROJO.append(np.median(grgb_1))
    curtosisDM_SF_HLROJO.append(np.cov(cropFourgb_1))
    nergianoglcmDM_SF_HLROJO.append(np.median(cropFourgb_1))

         
           
import pandas as pd    
datos = {'EnergiaDM_LH':energiaDM_LH,
         'HomogeneidadDM_LH':homogeneidadDM_LH,
         'CorrelaciónDM_LH':correlacionDM_LH,
         'DisimilitudDM_LH':disimiDM_LH,
         'ASM_DM_LH':ASMDM_LH,
         'EntropíaDM_LH':entropiaDM_LH,
         'ContrasteDM_LH':contrasteDM_LH,
         'Media(glcm)DM_LH':mediaglcmDM_LH,
         'MedianaDM_LH':modaDM_LH,
         'Entropia(no glcm)DM_LH':entropianoglcmDM_LH,
         'MediaDM_LH':mediaDM_LH,
         'DesviacionDM_LH':desviacionDM_LH,
         'covDM_LH':curtosisDM_LH,
         'Energia(no glcm)DM_LH':nergianoglcmDM_LH,         
         'EnergiaDM(sinF)_LH':energiaDM_SF_LH,
         'HomogeneidadDM(sinF)_LH':homogeneidadDM_SF_LH,
         'CorrelaciónDM(sinF)_LH':correlacionDM_SF_LH,
         'DisimilitudDM(sinF)_LH':disimiDM_SF_LH,
         'ASMDM(sinF)_LH':ASMDM_SF_LH,
         'EntropíaDM(sinF)_LH':entropiaDM_SF_LH,
         'ContrasteDM(sinF)_LH':contrasteDM_SF_LH,
         'Media(glcm)DM(sinF)_LH':mediaglcmDM_SF_LH,
         'MedianaDM_LH(sinF)':modaDM_SF_LH,
         'Entropia(no glcm)DM(sinF)_LH':entropianoglcmDM_SF_LH,
         'MediaDM(sinF)_LH':mediaDM_SF_LH,
         'DesviacionDM(sinF)_LH':desviacionDM_SF_LH,
         'COVDM(sinF)_LH':curtosisDM_SF_LH,
         'Energia(no glcm)DM(sinF)_LH':nergianoglcmDM_SF_LH,
         'EnergiaDM_HLROJO':energiaDM_HLROJO,
         'HomogeneidadDM_HLROJO':homogeneidadDM_HLROJO,
         'CorrelaciónDM_HLROJO':correlacionDM_HLROJO,
         'DisimilitudDM_HLROJO':disimiDM_HLROJO,
         'ASM_DM_HLROJO':ASMDM_HLROJO,
         'EntropíaDM_HLROJO':entropiaDM_HLROJO,
         'ContrasteDM_HLROJO':contrasteDM_HLROJO,
         'Media(glcm)DM_HLROJO':mediaglcmDM_HLROJO,
         'MEDIANADM_HLROJO':modaDM_HLROJO,
         'Entropia(no glcm)DM_HLROJO':entropianoglcmDM_HLROJO,
         'MediaDM_HLROJO':mediaDM_HLROJO,
         'DesviacionDM_HLROJO':desviacionDM_HLROJO,
         'CovDM_HLROJO':curtosisDM_HLROJO,
         'Energia(no glcm)DM_HLROJO':nergianoglcmDM_HLROJO,
         'EnergiaDM(sinF)_HLROJO':energiaDM_SF_HLROJO,
         'HomogeneidadDM(sinF)_HLROJO':homogeneidadDM_SF_HLROJO,
         'CorrelaciónDM(sinF)_HLROJO':correlacionDM_SF_HLROJO,
         'DisimilitudDM(sinF)_HLROJO':disimiDM_SF_HLROJO,
         'ASMDM(sinF)_HLROJO':ASMDM_SF_HLROJO,
         'EntropíaDM(sinF)_HLROJO':entropiaDM_SF_HLROJO,
         'ContrasteDM(sinF)_HLROJO':contrasteDM_SF_HLROJO,
         'Media(glcm)DM(sinF)_HLROJO':mediaglcmDM_SF_HLROJO,
         'medianaDM(sinF)_HLROJO':modaDM_SF_HLROJO,
         'Entropia(no glcm)DM(sinF)_HLROJO':entropianoglcmDM_SF_HLROJO,
         'MediaDM(sinF)_HLROJO':mediaDM_SF_HLROJO,
         'DesviacionDM(sinF)_HLROJO':desviacionDM_SF_HLROJO,
         'covDM(sinF)_HLROJO':curtosisDM_SF_HLROJO,
         'Energia(no glcm)DM(sinF)_HLROJO':nergianoglcmDM_SF_HLROJO}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasTW_NO.xlsx')        
 