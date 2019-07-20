# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:32:22 2019

@author: Nataly
"""
def comparacionhistRE(img,img2,segmenta,a):
    from readimg import read_img #leer imagines ### img=read_img(imgfile)##
    import cv2
    import pylab as plt 
    from matplotlib import pyplot as plt
    import numpy as np
    from numpy import asarray
    from PIL import Image
    from math import floor
    import pandas as pd
    import statistics as stats
    from scipy.spatial import distance
    import statistics
    imgseg=cv2.imread(segmenta,0)
    imgseg = cv2.normalize(imgseg, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    if a==0:
        for z in range((3)):
           img[:,:,z]=(img[:,:,z]*(imgseg))
        hista = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        imgseg=cv2.imread(segmenta,0)
        imgseg1=imgseg*-1
        imgseg1 = cv2.normalize(imgseg1, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        for z in range((3)):
           img2[:,:,z]=(img2[:,:,z]*(imgseg1))
        histb = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    else:
        img=img*imgseg
        hista = cv2.calcHist([img], [0], None, [256],[0, 256])
        imgseg=cv2.imread(segmenta,0)
        imgseg1=imgseg*-1
        imgseg1 = cv2.normalize(imgseg1, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        img2=img2*imgseg1
        histb = cv2.calcHist([img2], [0], None, [256],[0, 256])

#Euclidian Distance
    histograma = []
    histograma2 = []
    euclidian = []
    
    if a==0:
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            histograma.append(hist)
            histograma2.append(hist2)
    
        for l in range(3):
            eucl = distance.euclidean(histograma[l], histograma2[l])
            euclidian.append(eucl)
            #Dista euclidiana
            euclidiana = statistics.mean(euclidian)
    else:
         hist = cv2.calcHist([img], [0], None, [256], [0, 256])
         hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
         euclidiana = distance.euclidean(hist, hist2)
            
        
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2    
    #Correlación
    correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
    
    #Distancia Bhattacharyya
    Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA)
    
    #print('Correlación=', correlacion)
    #print('Distancia Bhattacharyya=', Bhattacharyya)
    return img,img2,hista,histb,correlacion,Bhattacharyya,euclidiana
