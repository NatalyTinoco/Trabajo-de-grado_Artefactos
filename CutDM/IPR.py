# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:21:18 2019

@author: Nataly
"""
import cv2
import numpy as np
import pylab as plt 
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

from skimage import feature
from skimage.morphology import skeletonize

def tloga(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img

def normalizacionMaxMin(img):
    img2 = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img2

import glob
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr

    
for file in glob.glob("*.jpg"): 
    img=cv2.imread(file)
    img=normalizacionMaxMin(img)
    #plt.imshow(img) 
    #plt.show()
    YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y,U,V=cv2.split(YUV)
    plt.imshow(V) 
    plt.show()
    # Calcution of Sobelx 
    sobelx = cv2.Sobel(V,cv2.CV_64F,1,0,ksize=5) 
    
    # Calculation of Sobely 
    sobely = cv2.Sobel(V,cv2.CV_64F,0,1,ksize=5) 
    sobelf = np.hypot(sobelx, sobely)
    sobelf *= 255.0 / np.max(sobelf)
    edge_roberts = roberts(V)
    edge_sobel = sobel(V)


    # Calculation of Laplacian 
    laplacian = cv2.Laplacian(V,cv2.CV_64F) 
    
    #canny 
    #img2 = cv2.normalize(V, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    min_ = 100
    max_ = 250
    th3 = cv2.adaptiveThreshold(V,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    edges = cv2.Canny(img,min_,max_,L2gradient=False) 
    #detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
    #edges = feature.canny(V)
    #edges = feature.canny(V, sigma=0.3)  
    #plt.imshow(sobel,'Greys') 
    #plt.show()
    #plt.imshow(laplacian,'Greys') 
    #plt.show()
    plt.imshow(edges*-1,'Greys') 
    plt.show()
    plt.imshow(th3,'Greys') 
    plt.show()
    #plt.imshow(edge_roberts,'Greys') 
    #plt.show()