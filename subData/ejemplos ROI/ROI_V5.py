# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:02:29 2019

@author: Nataly
"""

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
from pylab import *
#from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from skimage.morphology import disk

from skimage.filters import threshold_otsu
from skimage.filters import threshold_li
from skimage.filters import threshold_minimum
from skimage.filters import threshold_triangle
from skimage.filters import try_all_threshold
from PIL import ImageEnhance
from PIL import Image 
from skimage.filters import rank

import glob
def log(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img


for imgfile in glob.glob("*.jpg"):
    ima=cv2.imread(imgfile)
    #ima='00028_batch2.jpg'
    
    """ # Primera forma #"""
    #imR, imG, II=cv2.split(ima)
    """ #Segunda forma #"""
    #imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
    #I,I,II=cv2.split(imA)
    """ # Tercera forma #"""   
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2XYZ)
    #imA=cv2.cvtColor(ima,cv2.COLOR_RGB2YUV)
    I,I,II=cv2.split(imA)
    
    
    ta=II.shape
    ta=list(ta)
    #plt.imshow(II, cmap=plt.cm.gray)
    #plt.show()        
    MIN=int(np.min(II))
    MAX=int(np.max(II))
  
    hist = cv2.calcHist([II],[0],None,[MAX+1],[MIN,MAX])
    #hist = cv2.calcHist([II],[0],None,[256],[0,255])
    #plt.plot(hist)
    #plt.show()
    
    #div=8
    div=6
    nuevohist=hist.tolist() 
    l=int(len(nuevohist)/div)
    i=1
    a=0
    suma=np.zeros((div))
    valm=0
    hasta=0
    menor=0
    for y in range(1,div+1):
        suma[a]=sum(nuevohist[i:int(l)*y])
        if suma[a]>valm:
            valm=suma[a]
            hasta=y
        i=int(l)*y+1
        a=a+1
    porcen=0.3
    a=hist[:int((l*hasta)*porcen)]
    a=a.tolist()         
    hist=hist.tolist() 
    u=np.max(a)
    umbral1 = hist.index(u)
    a=a[umbral1:]
    
    uu=np.min(a)
    histb=hist[umbral1:]
    
    umbral=histb.index(uu)
    #print(umbral)
    umbral=umbral+len(hist[:umbral1])
    binary=II.copy()
    print(umbral)
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=umbral:
                binary[f,c]=0
            else:
                binary[f,c]=1

    binary = (binary*255).astype(np.uint8)
    #plt.imshow(binary, cmap=plt.cm.gray)
    #plt.show()
    #"""
    _,contours,_ = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    tru=30
    for f in range ((y+tru),(y+h-tru)):
        for c in range(x+tru,(x+w-tru)):
            #print(f,c)
            if binary[f+tru,c+tru]==1 or binary[f+tru,c]==1 or binary[f,c+tru]==1  and binary[f,c]==0:
                binary[f,c]=1
            else:
                binary[f,c]=binary[f,c]
    ### Transformaciones Morfologicas
    
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    openi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #37
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    close=cv2.morphologyEx(openi, cv2.MORPH_CLOSE, kernel)
    #"""
   
    #cv2.rectangle(ima,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imshow("Show",ima)
    #plt.imshow(im)
    #plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #"""
    #print((y,x),(y+h,x+w))
   

    #plt.imshow(close, cmap=plt.cm.gray)
    #plt.show()    
    #plt.imshow(openi, cmap=plt.cm.gray)
    #plt.show()

    dire='./segROI/#5/Z1/'+imgfile
    #cv2.imwrite(dire,close)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    print(imgfile)
