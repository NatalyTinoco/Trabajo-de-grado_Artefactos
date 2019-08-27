# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:28:48 2019

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
#from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')
#import matplotlib.patches as patches            
from skimage.measure import compare_ssim as ssim
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err    
#%%

for imgfile in glob.glob("*.jpg"):
#%%    
#    imgfile='00070_batch2.jpg'
#    imgfile='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/lena.jpg'
#    imgfile='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subRE/00201_batch2.jpg'
#    imgfile='00079.jpg'
#    imgfile='CT56_colitis_06697.jpg'
    imgfile='Flu_00461.jpg'
#    imgfile='00318.jpg'
#    imgfile='00054.jpg'
#    imgfile='0000592.jpg'
    img=cv2.imread(imgfile)   
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    taimg=img.copy()
    plt.imshow(img)
    plt.show()
    imaROI=ROI(img)
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
    if a>b:
        poten=math.log(a,2)
        poten=int(poten)
    else:
        poten=math.log(b,2)
        poten=int(poten)
    
    tama=2**poten
    aa = cv2.resize(V,(int(tama),int(tama)))
    HN=hadamard(tama, dtype=complex).real
    HT=np.transpose(HN)
    WHT=HN*aa*HT
    bb = cv2.GaussianBlur(aa,(5,5),2.5)
    WHTr=HT*bb*HT
    su=0
    sur=0
    p=0.76
    for f in range(tama):
        for c in range(tama):
            sur=abs(WHTr[f,c]**(p))+sur
            su=abs(WHT[f,c]**(p))+su
    sur=sur**(1/p)
    su=su**(1/p)
    
    beta1=sur/su
    print(beta1)
    print(sur-su)      
    ii=0            
    diferencia=[] 
    betanu=[]   
    error=[]
    mSSIM=[]         
    a,b=V.shape
    tamañoa1A=2**5
    tamañoa1B=2**5
    rea1=0
    Binary=V.copy()
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
       for ca1 in range(0,b-tamañoa1B,tamañoa1B):       
            croppeda1=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, V)
            aa = cv2.resize(croppeda1,( tamañoa1A, tamañoa1B))
#            plt.imshow(a,'Greys')
#            plt.show()
            HN=hadamard(tamañoa1A, dtype=complex).real
            HT=np.transpose(HN)
            WHT=HN*aa*HT
            bb = cv2.GaussianBlur(aa,(5,5),2.5)
#            bb=median(aa, disk(30))
            WHTr=HT*bb*HT
#            plt.imshow(WHTr,'Greys')
#            plt.show()
            su=0
            sur=0
            for f in range(tamañoa1A):
                for c in range(tamañoa1A):
                    sur=abs(WHTr[f,c]**2)+sur
                    su=abs(WHT[f,c]**2)+su
            sur=np.sqrt(sur)
            su=np.sqrt(su)
            betanu.append(sur/su)
            diferencia.append(su-sur)
            error.append(mse(WHTr,WHT))
            mSSIM.append(ssim(WHTr,WHT))
            print('beta',sur/su)
            print('diferencia',su-sur) 
            ii=ii+1
            print(ii)

#%%
    X=np.array(list(zip(diferencia,error)))
    
    kmeans=KMeans(n_clusters=2)
    kmeans=kmeans.fit(X)
    labels=kmeans.predict(X)
    centroids=kmeans.cluster_centers_
    
    colors=["m.","r.",".c","y.","b."]
    for i in range(len(X)):
        print('Coordenada: ',X[i],'Etiqueta: ',labels[i])
        plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
    
    plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=150,linewidths=5,zorder=10)
    plt.show()
    
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
            Binary[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B]=labels[o]
            if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:
               if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
                     croppeda1= V[fa1:a,ca1:b]
                     ta1,ta2=croppeda1.shape
                     Binary[fa1:a,ca1:b]=labels[o]
               else:
                      croppeda1 = V[fa1:fa1+tamañoa1A,ca1:]
                      Binary[fa1:fa1+tamañoa1A,ca1:]=labels[o]
            if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
#                 print('ola')
                 if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:
#                     print(fa1)
                     croppeda1 = V[fa1:a,ca1:b]
                     Binary[fa1:a,ca1:b]=labels[o]
                     
                 else:
                     croppeda1 = V[fa1:,ca1:ca1+tamañoa1B] 
                     Binary[fa1:,ca1:ca1+tamañoa1B]=labels[o]
            o=o+1
            print('VALOR O', o)
    fila,Col,cha=taimg.shape
    Binaryfinal=np.zeros((fila,Col)).astype(np.uint8)
    Binaryfinal[y:y+h,x:x+w]=Binary*255
    #%%
    cv2.imshow('image',Binaryfinal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    