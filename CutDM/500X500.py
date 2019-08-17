# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:59:00 2019

@author: Nataly
"""


import cv2
import numpy as np
import glob
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from scipy.stats import kurtosis
import statistics as stats
import pywt
import pywt.data



dm=0
re=0
no=0
    
for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    #cv2.imshow('Grays',imaROI)
    #cv2.destroyAllWindows()
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
    #print(image,dm)
    a,b= V.shape
    #tamañoA,tamañoB=tama(a,b)
    V1= V
    imdm= im
    tamañoA=500
    tamañoB=500
    vecesA = int(a/tamañoA)
    vecesB = int(b/tamañoB)
    
    for f in range(0,a-tamañoA,tamañoA):
        for c in range(0,b-tamañoB,tamañoB):
            #print(f,c)
            cropped = V1[f:f+tamañoA,c:c+tamañoB]
            croppedrgb = imdm[f:f+tamañoA,c:c+tamañoB]
           
            #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
            if c==tamañoB*vecesB-tamañoB:
                cropped = V1[f:f+tamañoA,c:]
                croppedrgb = imdm[f:f+tamañoA,c:]
                #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
            if f==tamañoA*vecesA-tamañoA:
                 #print('ola')
                 if c==tamañoB*vecesB-tamañoB:
                    cropped = V1[f:,c:]
                    croppedrgb = im[f:,c:]
               
                     #test2[f:,c:]=test[f:,c:]
                 else:
                     cropped = V1[f:,c:c+tamañoB]
                     croppedrgb = imdm[f:,c:c+tamañoB]
               
                     #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                     #print('dani')
            dm=dm+1  
            cropped=cv2.resize(cropped,(500,500))      
            croppedrgb=cv2.resize(croppedrgb,(500,500))  
            diredm='./500X500/DM/color/'+str(dm)+image
            cv2.imwrite(diredm,croppedrgb)
            diredm2='./500X500/DM/V/'+str(dm)+image
            cv2.imwrite(diredm2,cropped)
            #print('DM',dm)
#                    plt.imshow(croppedrgb)
#                    plt.show()