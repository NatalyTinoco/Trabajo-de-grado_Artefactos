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
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#%%

for imgfile in glob.glob("*.jpg"):
#    imgfile='00070_batch2.jpg'
#    imgfile='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/lena.jpg'
#    imgfile='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subRE/00201_batch2.jpg'
#    imgfile='00079.jpg'
#    imgfile='CT56_colitis_06697.jpg'
    imgfile='00318.jpg'
#    imgfile='00054.jpg'
#    imgfile='0000592.jpg'
    img=cv2.imread(imgfile)   
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    plt.imshow(img)
    plt.show()
    imaROI=ROI(img)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range(3):
        img[:,:,z]=img[:,:,z]*imaROI
    plt.imshow(img)
    plt.show()
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    a,b,ch = img[y:y+h,x:x+w].shape
    img=img[y:y+h,x:x+w]
    
    
    HSV=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
#    V=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#    H,S,V=cv2.split(HSV)
    if a>b:
        poten=math.log(a,2)
        poten=int(poten)
    else:
        poten=math.log(b,2)
        poten=int(poten)
    
    tama=2**poten
    aa = cv2.resize(V,(int(tama),int(tama)))
    plt.imshow(aa,'Greys')
    plt.show()
    HN=hadamard(tama, dtype=complex).real
    HT=np.transpose(HN)
    WHT=HN*aa*HT
#    WH=np.dot(HN,a)
#    WHT=np.dot(WH,HT)
    plt.imshow(WHT,'Greys')
    plt.show()
    
    bb = cv2.GaussianBlur(aa,(5,5),2.5)
#    b=median(a, disk(30))
    plt.imshow(bb,'Greys')
    plt.show()
#    b=median(a, disk(20))
    WHTr=HT*bb*HT
    plt.imshow(WHTr,'Greys')
    plt.show()
    
    su=0
    sur=0
#    p=2
    p=0.76
    for f in range(tama):
        for c in range(tama):
            sur=abs(WHTr[f,c]**(p))+sur
            su=abs(WHT[f,c]**(p))+su
#    sur=np.sqrt(sur)
#    su=np.sqrt(su)
    sur=sur**(1/p)
    su=su**(1/p)
    
    beta1=sur/su
    print(beta1)
    #obeta=norm(WHTr)/norm(WHT)
    print(sur-su)      
 #%%  
    #%% RUTA 2
#    nueva,tamal,tamanu=agregarceros(V,11)
#  
#    mbeta=np.zeros((tamal[0],tamal[1]))
#    ss=np.zeros((tamal[0],tamal[1]))
#    ta=mbeta.shape
#    ta=list(ta)
#    corte=5
#    for x in range (corte,tamanu[0]-corte) :
#        for y in range(corte,tamanu[1]-corte):
#            nu=nueva[f-corte:f+corte,c-corte:c+corte]
#            a = cv2.resize(nueva,(int(8),int(8)))
##            plt.imshow(a,'Greys')
##            plt.show()
#            HN=hadamard(8, dtype=complex).real
#            HT=np.transpose(HN)
#            WHT=HN*a
#            b = cv2.GaussianBlur(a,(5,5),2.5)
##            b=median(a, disk(30))
#            WHTr=HT*b
##            plt.imshow(WHTr,'Greys')
##            plt.show()
#            su=0
#            sur=0
#            for f in range(8):
#                for c in range(8):
#                    sur=abs(WHTr[f,c]**2)+sur
#                    su=abs(WHT[f,c]**2)+su
#            sur=np.sqrt(sur)
#            su=np.sqrt(su)
#            
#            betanu=sur/su
##            print(beta1)
##            #obeta=norm(WHTr)/norm(WHT)
##            print(su-sur) 
#            mbeta[x-corte,y-corte]=betanu
#            ss[f-corte:f+corte,c-corte:c+corte]=V[f-corte:f+corte,c-corte:c+corte]
#    plt.imshow(ss,'Greys')
#    plt.show()
#    #mbeta=mbeta.astype(np.float64)
#    plt.imshow(mbeta,'Greys')
#    plt.show()
#    mbeta = cv2.normalize(mbeta, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#    plt.imshow(mbeta)
#    plt.show()
#%%
    tamañoa1A=2**7
    tamañoa1B=2**7
    filetxt=imgfile[0:len(imgfile)-3]+'txt'
#    f=open(datafolder/filetxt)
#          
#    bboxfile=f.read()
    boxes = read_boxes(filetxt)
    
    boxes_abs = yolo2voc(boxes, img.shape)  
    for b in boxes_abs:
            cls, x1, y1, x2, y2 = b
            if cls == 3:
                cropped=V[int(y2):int(y2+40),int(x2):int(x2+40)]
                aa = cv2.resize(cropped,( tamañoa1A, tamañoa1B))
    #            plt.imshow(a,'Greys')
    #            plt.show()
                HN=hadamard(tamañoa1A, dtype=complex).real
                HT=np.transpose(HN)
                WHT=HN*aa*HT
#                bb = cv2.GaussianBlur(aa,(5,5),2.5)
                bb=median(aa, disk(30))
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
                
                print(sur/su)
                print(sur-su)
                cv2.rectangle(img,(int(x1),int(y1)),(int(x1)+int(x2),int(y1)+int(y2)),(255,0,0),2)
                cv2.imshow('image',img)
                
#%%  
    import matplotlib.patches as patches            
    from skimage.measure import compare_ssim as ssim
    def mse(imageA, imageB):
    	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    	err /= float(imageA.shape[0] * imageA.shape[1])
    	return err    
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
#            croppedrgb=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, img)
#            rgb=cv2.cvtColor(croppedrgb,cv2.COLOR_RGB2BGR)
#            plt.imshow(rgb)
#            plt.show()  
            print('beta',sur/su)
#            #obeta=norm(WHTr)/norm(WHT)
            print('diferencia',su-sur) 
            ii=ii+1
#            print(ii)
#              #print(f,c)
#            vecesA = int(a/tamañoa1A)
#            vecesB = int(b/tamañoa1B)  
#            if ca1<tamañoa1B*vecesB-tamañoa1B and fa1<tamañoa1A*vecesA-tamañoa1A:
#                cv2.rectangle(img,(fa1,ca1),(fa1+tamañoa1A,ca1+tamañoa1B),(0,255,0),2)
#                cv2.imshow('image',img)
##                print(fa1,fa1+tamañoa1A,'=',ca1,ca1+tamañoa1B)
#                #Binary[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B] = binary
#                  #x,y,w,h = cv2.boundingRect(cnt)
#                  #a,b,ch = img[y:y+h,x:x+w].shape
#            if ca1==tamañoa1B*vecesB-tamañoa1B and fa1<tamañoa1A*vecesA-tamañoa1A:
#                cv2.rectangle(img,(fa1,ca1),(fa1+tamañoa1A,ca1+b),(0,255,0),2)
#                cv2.imshow('image',img)
#              # print('paso')
###               Binary[fa1:fa1+tamañoa1A,ca1:]=binary
#            if fa1==tamañoa1A*vecesA-tamañoa1A:
#               if ca1==tamañoa1B*vecesB-tamañoa1B:
#                    cv2.rectangle(img,(fa1,ca1),(a,b),(0,255,0),2)
#                    cv2.imshow('image',img)   
###                  Binary[fa1:,ca1:]=binary
#               else:
#                    cv2.rectangle(img,(fa1,ca1),(a,ca1+tamañoa1B),(0,255,0),2)
#                    cv2.imshow('image',img)
##                  Binary[fa1:,ca1:ca1+tamañoa1B]=binary
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()        
    
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from collections import defaultdict
import matplotlib.cm as cm
 
X=np.zeros((len(diferencia),2))
X[:,0]=diferencia
X[:,1]=error
m=X.shape[0]
n_iter=100

#%%
from Kmeans import Kmeans
f,axarr=plt.subplots(5,2,figsize=(15,30))
i=0
j=0
#WCSS_array=np.array([])
#for K in range(1,11):
#    kmeans=Kmeans(X,K)
#    kmeans.fit(n_iter)
#    Output,Centroids=kmeans.predict()
#    wcss=0
#    for k in range(K):
#        wcss+=np.sum((Output[k+1]-Centroids[k,:])**2)
#    WCSS_array=np.append(WCSS_array,wcss)
#    for k in range(K):
#        axarr[i,j].scatter(Output[k+1][:,0],Output[k+1][:,1])
#    axarr[i,j].scatter(Centroids[:,0],Centroids[:,1],s=300,c='yellow',label='Centroids')
#    axarr[i,j].set_title('Clustered data with '+str(K)+' clusters')
#    if(K%2==1):
#        j+=1
#    else:
#        j=0
#        i+=1
#for ax in axarr.flat:
#    ax.set(xlabel='Income', ylabel='Number of transactions')
#    
#    #WCSS_array=np.append(WCSS_array,kmeans.WCSS())
#    
#K_array=np.arange(1,11,1)
#plt.plot(K_array,WCSS_array)
#plt.xlabel('Number of Clusters')
#plt.ylabel('within-cluster sums of squares (WCSS)')
#plt.title('Elbow method to determine optimum number of clusters')
#plt.show()
#%%
K=2



kmeans=Kmeans(X,K)
kmeans.fit(n_iter)
Output,Centroids=kmeans.predict()

color=['red','blue']
labels=['cluster1','cluster2']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    print(k)
    
plt.scatter(Centroids[:,0],Centroids[:,1],s=600,c='yellow',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()



    