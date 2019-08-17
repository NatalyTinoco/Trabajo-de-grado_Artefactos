# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 01:04:42 2019

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

def SingularValueFeature(A):
    #import numpy.linalg as svd 
    k,k1=A.shape
    U,s,V=np.linalg.svd(A,full_matrices=False)
    #print(U.shape,s.shape,V.shape)
    reconst_matrix=np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return  reconst_matrix,s

contrast=[]
energi=[]
homogenei=[]
correlaci=[]
disi=[]
AS=[]
entrop=[]

contrast1=[]
energi1=[]
homogenei1=[]
correlaci1=[]
disi1=[]
AS1=[]
entrop1=[] 

beta=[]
sumas=[]
media=[]
mediana=[]
destan=[]
var=[]
correla=[]
covar=[]    
 
#from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    aa,bb,c = im.shape    
    croppedrgb=im
    #V=cropped
    cropped=cv2.imread('./V/'+image,0)
    print(cropped.shape)
    B,T=SingularValueFeature(cropped)
    T=T.tolist() 
    u=np.max(T)
    TT= T.index(u)
    if TT==0:
       betadm=T[TT]/sum(T)
    else:
        betadm=sum(T[0:TT])/sum(T)
    beta.append(betadm)
    sumas.append(sum(T))
    media.append(np.mean(T))
    mediana.append(np.median(T))
    destan.append(np.std(T))
    var.append(np.var(T))
    correla.append(np.corrcoef(T))
    covar.append(np.cov(T))         
    
    cropFou=Fourier(B)
    g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(cropFou)
    contrast.append(contraste)
    energi.append(energia)
    homogenei.append(homogeneidad)
    correlaci.append(correlacion)
    disi.append(disimi)
    AS.append(ASM)
    entrop.append(entropia)

   
import pandas as pd    
datos = {'Contraste':contrast,
         'Energia':energi,
         'Homogeneidad':homogenei,
         'Correlación':correlaci,
         'Disimilitud':disi,
         'ASM':AS,
         'Entropia':entrop,
         'beta':beta,
         'sumas':sumas,
         'media':media,
         'mediana':mediana,
         'desviacion E':destan,
         'Varianza':var,
         'Corralación':correla,
         'Covarianza':covar}
datos = pd.DataFrame(datos)
datos.to_excel('SVDRE.xlsx') 

