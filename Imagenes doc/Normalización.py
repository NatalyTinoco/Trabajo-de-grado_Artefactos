# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:55:18 2019
"""
import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
from math import floor
import pandas as pd
import statistics as stats

def normalizacionMaxMin(img):
    import cv2
    img2 = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img2
#img2 = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

def normalizacionrgb(ima):
    ima=np.asarray(ima)
    imR=ima[:,:,2]
    imG=ima[:,:,1]
    imB=ima[:,:,0]
    r=ima.shape
    r=list(r)
    imb=np.zeros((r[0],r[1]))
    img=np.zeros((r[0],r[1]))
    imr=np.zeros((r[0],r[1]))
    for f in range(0, r[0]):
        for c in range(0,r[1]) :
                sumaRGB=float (imR[f,c]+imG[f,c]+imB[f,c])
                imb[f,c]=imB[f,c]/sumaRGB
                img[f,c]=imG[f,c]/sumaRGB
                imr[f,c]=imR[f,c]/sumaRGB
    
    ima[:,:,0]=imb
    ima[:,:,1]=img
    ima[:,:,2]=imr  
    return ima
def normalizacionintensi(img):
    v=19
    c=0
    cc=0
    vv=19
    tamanu=img.shape
    tamanu=list(tamanu)
    #ima=img
    #print(tamanu)
    #print(tamanu[0]/20,int (tamanu[1]/20))
    for x in range (2,int(tamanu[0]/20)+3) :
        for y in range(2,int(tamanu[1]/20)+3):
            #print('for',x,y)
            factor=0.05
            mb=img[c:v,cc:vv,0].tolist()
            mb=np.hstack(mb)
            medianaB=stats.median(mb)*factor
            
            mg=img[c:v,cc:vv,0].tolist()
            mg=np.hstack(mg)
            medianaG=stats.median(mg)*factor
            
            mr=img[c:v,cc:vv,2].tolist()
            mr=np.hstack(mr)
            medianaR=stats.median(mr)*factor
            #print(medianaB,medianaG,medianaR)
            
            img[c:v,cc:vv,0]= img[c:v,cc:vv,0]-medianaB
            img[c:v,cc:vv,1]= img[c:v,cc:vv,1]-medianaG
            img[c:v,cc:vv,2]= img[c:v,cc:vv,2]-medianaR
            #print('po',c,v,'o',cc,vv)
            #plt.imshow(img[c:v,cc:vv,:])
            #plt.show()
            vv=19*y
            cc=cc+19 
        v=19*x
        c=c+19
        vv=19
        cc=0
    return img

def estandarizacion(img):
    import numpy as np
    mean = np.mean(img, axis=(1,2), keepdims=True)
    std = np.std(img, axis=(1,2), keepdims=True)
    img = (img - mean) / std
    return img
def tloga(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img

"""
#file='Lap_01004'
#file='WL_00485'
file='00000'
#file='Lap_01004'
#file='test'
#file='rgb (1)'
#file='NCL (532)'file='Lap_01004'
#file='DB58_colitis_full_00479'
#file='DB58_colitis_full_03038'
#file='gastro_00377'
#file='CT56_colitis_02461'
#file='Flu_01025'

img=cv2.imread(file+'.jpg')
img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
plt.imshow(img)
plt.show()
print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
plt.hist(img.ravel(),256,[img.min(),img.max()]); plt.show()
img=normalizacionintensi(img)
# Apply log transform
plt.imshow(img)
plt.show()  
 
tamanu=img.shape
print(tamanu)
print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
plt.hist(img.ravel(),256,[img.min(),img.max()]); plt.show()
#plt.imshow(img[0:19,0:19,:])
#plt.show()
#"""
