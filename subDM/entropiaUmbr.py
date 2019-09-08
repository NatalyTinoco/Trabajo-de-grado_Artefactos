# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:58:49 2019

@author: Nataly
"""
import cv2 
import glob 
import sys
import numpy as np
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from yolovoc import yolo2voc
from readboxes import read_boxes
from rOI import ROI
from ventaneo import ventaneoo
from agregaceros import agregarceros
from filtromedianaa import filtromediana

from matplotlib import pyplot as plt



import pylab as py


from scipy.stats import linregress
def filtrogaussiano(img):
    B,G,R=cv2.split(img)
    fB=cv2.GaussianBlur(B,(3,3),0)
    fR=cv2.GaussianBlur(R,(3,3),0)
    fG=cv2.GaussianBlur(G,(3,3),0)
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh
from math import log
import numpy as np
from time import clock

def de2bi(d, n):
    d = np.array(d)
    power = 2**np.arange(n)
    d = d * np.ones((1,n))
    b = np.floor((d%(2*power))/power)
    return b
from scipy.linalg import hadamard
from numpy.linalg import norm
from skimage.filters.rank import median
from skimage.morphology import disk
import math


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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
#from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')
#import matplotlib.patches as patches            
from skimage.measure import compare_ssim as ssim
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err  

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



#import radialProfile
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

import pylab as py
def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]
#%%
for imgfile in glob.glob("*.jpg"):  
    #%%
#    imgfile='00070_batch2.jpg'
#    imgfile='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/lena.jpg'
#    imgfile='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subRE/00201_batch2.jpg'
#    imgfile='00079.jpg'
#    imgfile='CT56_colitis_06697.jpg'
#    imgfile='00007.jpg'
#    imgfile='00318.jpg'
#    imgfile='00054.jpg'
#    imgfile='00064_batch2.jpg'
#    imgfile='00085.jpg'

#    imgfile='00119_batch2.jpg'
        
    img=cv2.imread(imgfile)  
    imaROI=ROI(img)
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    taimg=img.copy()
#    plt.imshow(img)
#    plt.show()
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range(3):
        img[:,:,z]=img[:,:,z]*imaROI
#    plt.imshow(img)
#    plt.show()
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    a,b,ch = img[y:y+h,x:x+w].shape
    img=img[y:y+h,x:x+w]
    
    
    HSV=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
    a,b=S.shape
    aa=S.copy()
    bb = cv2.medianBlur(aa, 15)
    
    img_back=DFT(V)
    j=azimuthalAverage(img_back)
    xs=np.arange(len(j))
    slope = linregress(xs, j)[0]  # slope in units of y / x
    slope_angle = math.atan(slope)  # slope angle in radians
    alfa0 = math.degrees(slope_angle) 
    p1=[]
    alfaplista=[]
    ii=0            
    diferencia=[] 
    betanu=[]   
    error=[]
    mSSIM=[]         
    contrast=[]
    energi=[]
    homogenei=[]
    correlaci=[]
    disi=[]
    AS=[]
    entrop=[]  
    nuevoindice1=[]
    nuevoindice2=[]
    
    tamañoa1A=100
    tamañoa1B=100
    rea1=0
    Binary=V.copy()
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
       for ca1 in range(0,b-tamañoa1B,tamañoa1B):       
            croppeda1=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, S)
#            plt.imshow(croppeda1,'Greys')
#            plt.show()
            aac = croppeda1.copy()
            bbc =  cv2.medianBlur(aa, 15)
            su=0
            sur=0
            for f in range(tamañoa1A):
                for c in range(tamañoa1A):
                    sur=abs(bbc[f,c]**2)+sur
                    su=abs(aac[f,c]**2)+su
            sur=np.sqrt(sur)
            su=np.sqrt(su)
#            betanu.append(sur/su)
            diferencia.append(su-sur)
#            betaimacom1=int((sur/su)/(betaimacom))
#            diferenciaimacom2=int((su-sur)/(diferenciaimacom))
#            print(betaimacom1)
#            nuevoindice1.append(betaimacom1)
#            nuevoindice2.append(diferenciaimacom2)
            error.append(mse(aa,bb))
            mSSIM.append(ssim(bb,aa))
            cropFou=Fourier(croppeda1)
            g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(cropFou)
            contrast.append(contraste)
            energi.append(energia)
            
            img_backc=DFT(croppeda1)
            jc=azimuthalAverage(img_backc)
            gradientec=np.gradient(jc)
            alfapma=np.max(gradientec)
            xsc=np.arange(len(jc))
            slopec = linregress(xsc, jc)[0]  # slope in units of y / x
            slope_anglec = math.atan(slopec)  # slope angle in radians
            alfap = math.degrees(slope_anglec) 
            p1.append((alfap-alfa0)/alfa0)
            alfaplista.append(alfap)
            #homogenei.append(homogeneidad)
            #correlaci.append(correlacion)
            #disi.append(disimi)
            #AS.append(ASM)
            entrop.append(entropia)
            ii=ii+1
#            print(ii)
#%%
#    X=np.array(list(zip(entrop,mSSIM,energi,AS)))
#    X=np.array(list(zip(mSSIM,diferencia)))  #MAS O MENOS
    X=np.array(list(zip(p1,mSSIM)))      
#    X=np.array(list(zip(contrast,homogenei)))        
#    X=np.array(list(zip(disi,mSSIM)))   # NOOO        
#    X=np.array(list(zip(entrop,mSSIM))) #BIEN
#    X=np.array(list(zip(entrop,diferencia))) #BI
#    X=np.array(list(zip(diferencia,energi)))   ##no da tan mal   umb1         
    kmeans=KMeans(n_clusters=2)
    kmeans=kmeans.fit(X)
    labels=kmeans.predict(X)
    centroids=kmeans.cluster_centers_
#    colors=["m.","r.",".c","y.","b."]
#    for i in range(len(X)):
#        print('Coordenada: ',X[i],'Etiqueta: ',labels[i])
#        plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
#    plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=150,linewidths=5,zorder=10)
#    plt.show()
#%%
    labels2=labels.copy()
    Binary=S.copy()
    o=0
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
       for ca1 in range(0,b-tamañoa1B,tamañoa1B):       
            vecesA = int(a/tamañoa1A)
            vecesB = int(b/tamañoa1B)
            croppeda1 = V[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B]
#            croppeda1rgb = img[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B,:]
            ta1,ta2=croppeda1.shape
            binary=croppeda1.copy()
#            for ff in range(ta1):
#                   for cc in range (ta2):
#                       if labels[o]==0:
#                           binary[ff,cc]=0
#                       else:
#                           binary[ff,cc]=255
#                           print('BLANCO')
            Binary[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B]=labels2[o]
            if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:
               if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
                     croppeda1= V[fa1:a,ca1:b]
                     ta1,ta2=croppeda1.shape
                     Binary[fa1:a,ca1:b]=labels2[o]
               else:
                      croppeda1 = V[fa1:fa1+tamañoa1A,ca1:]
                      Binary[fa1:fa1+tamañoa1A,ca1:]=labels2[o]
            if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
#                 print('ola')
                 if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:
#                     print(fa1)
                     croppeda1 = V[fa1:a,ca1:b]
                     Binary[fa1:a,ca1:b]=labels2[o]
                     
                 else:
                     croppeda1 = V[fa1:,ca1:ca1+tamañoa1B] 
                     Binary[fa1:,ca1:ca1+tamañoa1B]=labels2[o]
            o=o+1
#            print('VALOR O', o)
    fila,Col,cha=taimg.shape
    Binaryfinal=np.zeros((fila,Col)).astype(np.uint8)
    for hh in range(Binary.shape[0]):
        for hc in range(Binary.shape[1]):
            if Binary[hh,hc]==0:
                Binary[hh,hc]=1
            else:
                Binary[hh,hc]=0
    
    if Binary.shape==Binaryfinal.shape:
        Binaryfinal=Binary
    else:
       Binaryfinal[y:y+h,x:x+w]=Binary 
    Binaryfinal=Binaryfinal*imaROI 
    Binaryfinal=Binaryfinal*255
    dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subDM/umb_DM_NUEVA2/'+imgfile
    cv2.imwrite(dire,Binaryfinal)

#    Binaryfinal= cv2.normalize(Binaryfinal, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#    cv2.imshow('image',Binaryfinal)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()