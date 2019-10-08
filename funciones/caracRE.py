# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 09:41:30 2019

@author: Usuario
"""
import cv2
import numpy as np
import skimage.feature

brillomedia=[]
desviacionB=[]
contrastB=[]

def GLCM (imA):
    a=int(np.max(imA))
    g = skimage.feature.greycomatrix(imA, [1], [0], levels=a+1, symmetric=False, normed=True)                  
    contraste=skimage.feature.greycoprops(g, 'contrast')[0][0]
    return contraste

def caracRe(imabbox):
    R,G,B=cv2.split(imabbox)
    width,heigth,ch=imabbox.shape
    
    brillo=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(width*heigth)
    brillomedia.append(np.mean(brillo))
    
    contrastB.append(GLCM(B))
    desviacionB.append(np.std(B))
    
    return np.mean(brillo),GLCM(B),np.std(B)