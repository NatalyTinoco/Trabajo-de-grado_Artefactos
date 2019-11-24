# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:52:08 2019

@author: Nataly
"""
from matplotlib import pyplot as plt
import cv2 
from skimage.measure import compare_ssim as ssim
from skimage.filters.rank import median
from skimage.morphology import disk
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from Filtros import filtromediana

file=['00000.jpg','00064.jpg']
for i in file:
    img = cv2.imread(i)
    imga=img.copy()
    img1=cv2.cvtColor(imga, cv2.COLOR_RGB2BGR)  
    plt.imshow(img1)
    plt.show()
    nu=15
    bb4= filtromediana(imga,nu)
    bb44=cv2.cvtColor(bb4, cv2.COLOR_RGB2BGR) 
    plt.imshow(bb44)
    plt.show()       
    SSIMN=ssim(bb4,img,multichannel=True)
    print(SSIMN)
    