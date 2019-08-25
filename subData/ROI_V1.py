# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:37:26 2019

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
import glob

for imgfile in glob.glob("*.jpg"):
    imgfile='00002.jpg'
#    imgfile='CT56_colitis_02446.jpg'
    ima=cv2.imread(imgfile)
    
    """ # Primera forma #"""
#    imR, imG, II=cv2.split(ima)
    """ #Segunda forma #"""
#    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2HSV)
#    I,I,II=cv2.split(imA)
    """ # Tercera forma #"""   
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2XYZ)
    I,I,II=cv2.split(imA)
    plt.imshow(II, cmap=plt.cm.gray)
    plt.show()
    hist = cv2.calcHist([II],[0],None,[256],[0,255])
    plt.plot(hist)
    plt.show()
    
    
    ta=II.shape
    ta=list(ta)
    binary=II.copy()
    for f in range(ta[0]):
        for c in range (ta[1]):
            if II[f,c]<=0.03:
                binary[f,c]=0
            else:
                binary[f,c]=1
                    
    binary = (binary*255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    #plt.imshow(opening, cmap=plt.cm.gray)
    #plt.show()
#   
#    dire='./segROI/#1/B/'+imgfile
#    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
#    cv2.imwrite(dire,close)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    plt.imshow(close,'Greys')
    plt.show() 
    
    ROI = cv2.normalize(close, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range (3):
        ima[:,:,z]=ima[:,:,z]*ROI
    ima=cv2.cvtColor(ima,cv2.COLOR_RGB2BGR)
    plt.imshow(ima)
    plt.show() 
    
#datos = pd.DataFrame(datos)
##datos.to_excel('GLCMRES.xlsx') 
##datos.to_excel('GLCMRE.xlsx')
#datos.to_excel('GLCMREY.xlsx')
