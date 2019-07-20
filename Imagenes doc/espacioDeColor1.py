# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:04:37 2019

@author: Nataly
"""

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
from scipy import stats
from math import floor
import matplotlib.pyplot
import pylab

def histograma3(img):
    for i, c in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.show()
    return []

def grafica3color(img,R,G,B,nombre):
     
    fig, ax = plt.subplots(ncols=4, figsize=(12, 7), sharex=True, sharey=True)
    ax[0].imshow(img)
    ax[0].set_title(nombre[0])
    ax[1].imshow(R,cmap=plt.cm.gray)
    ax[1].set_title(nombre[1])
    ax[2].imshow(G,cmap=plt.cm.gray)
    ax[2].set_title(nombre[2])
    ax[3].imshow(B,cmap=plt.cm.gray)
    ax[3].set_title(nombre[3])
    return []
from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from matplotlib import pyplot as plt
import glob
from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio
i=0
for file in glob.glob("*.jpg"): 
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=filtrominimo(img)
    B,G,R=cv2.split(img)
 
    plt.imshow(img)
    plt.show()
    color = ('b','g','r')
    #histograma3(img)
    
    HSI=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,I=cv2.split(HSI)
    #histograma3(HSI)
    
    YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y,U,V=cv2.split(YUV)
    #histograma3(YUV)
    
    xyz=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
    x,y,z=cv2.split(xyz)
    #histograma3(xyz)
    
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    l,a,b=cv2.split(lab)
    #histograma3(lab)
    
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    yc,cr,cb=cv2.split(ycrcb)
    #histograma3(ycrcb)
    
    UNO=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    U,N,O=cv2.split(UNO)
   # histograma3(UNO)
    
    cv2.destroyAllWindows()
    
    nombre=('RGB','R','G','B')
    grafica3color(img,R,G,B,nombre)
    nombre=('HSV','H','S','V')
    grafica3color(HSI,H,S,I,nombre)
    nombre=('YUV','Y','U','V')
    grafica3color(YUV,Y,U,V,nombre)
    nombre=('XYZ','X','Y','Z')
    grafica3color(xyz,x,y,z,nombre)
    nombre=('Lab','L','a','b')
    grafica3color(lab,l,a,b,nombre)
    nombre=('YCrCb','Y','Cr','Cb')
    grafica3color(ycrcb,yc,cr,cb,nombre)
    nombre=('HLS',' H',' L','S')
    grafica3color(UNO,U,N,O,nombre)
    
    
    
    
    """
    data = (B, G, R)
    colors = ("blue", "green", "red")
    groups = ("B", "G", "R")
    r=img.shape
    r=list(r)
    #plt.scatter(R,G, alpha=0.8, c='red', edgecolors='none', s=30, label='Red')
    for f in range(r[0]):
        for c in range (r[1]):    
            #for data, color, group in zip(data, colors, groups):
                #x, y = data
                plt.scatter(R[f,c],f, alpha=0.8, c='red', edgecolors='none', s=30, label='Red')
    
    """