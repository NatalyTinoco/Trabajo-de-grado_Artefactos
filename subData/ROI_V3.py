# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:32:38 2019

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
from skimage.filters import threshold_otsu, threshold_local

from PIL import ImageEnhance
from PIL import Image 
import glob
from skimage.segmentation import flood, flood_fill
umbral=[]
for imgfile in glob.glob("*.jpg"):
#    imgfile='00002.jpg'
#    imgfile='CT56_colitis_02446.jpg'
    ima=cv2.imread(imgfile)
    """ # Primera forma #"""
#    imR, imG, II=cv2.split(ima)
    """ #Segunda forma #"""
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
    I,I,II=cv2.split(imA)
    """ # Tercera forma #"""   
    #imA=cv2.cvtColor(ima,cv2.COLOR_RGB2XYZ)
    #I,I,II=cv2.split(imA)
#    plt.imshow(II, cmap=plt.cm.gray)
#    plt.show() 
    #fig, ax = try_all_threshold(II, figsize=(10, 8), verbose=False)
    #plt.show()  
    #block_size = 51
    #thresh = threshold_local(II, block_size, offset=10)
    thresh = threshold_li(II)
    #thresh=threshold_minimum(II)
    #thresh=threshold_otsu(II)
    #thresh=threshold_triangle(II)
    #ret3,thresh= cv2.threshold(II,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(thresh)
    umbral.append(thresh)
#    binary = II > thresh
#    #"""
#                    
#    binary = (binary*255).astype(np.uint8)
#    
#    #plt.imshow(binary, cmap=plt.cm.gray)
#    #plt.show()
#    hist = cv2.calcHist([II],[0],None,[256],[0,255])
#    plt.plot(hist)
#    plt.show()
#    
#
#    #"""
#    ### Transformaciones Morfologicas
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
#    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#    #dilation = cv2.dilate(opening,kernel,iterations = 1)
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
#    close=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#
#    #plt.imshow(close, cmap=plt.cm.gray)
#    #plt.show()
##   
##    dire='./segROI/#3/B/'+imgfile
##    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
##    cv2.imwrite(dire,close)
##    print(imgfile)
##    k = cv2.waitKey(1000)
##    #destroy the window
##    cv2.destroyAllWindows()
#    #"""
#    plt.imshow(close,'Greys')
#    plt.show() 
#    
##    ROI = cv2.normalize(close, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
##    for z in range (3):
##        ima[:,:,z]=ima[:,:,z]*ROI
##    ima=cv2.cvtColor(ima,cv2.COLOR_RGB2BGR)
##    plt.imshow(ima)
##    plt.show() 
import pandas as pd    
datos = {'umbral':umbral}
datos = pd.DataFrame(datos)
datos.to_excel('umbralROI_2.xlsx')
     