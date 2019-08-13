# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:35:12 2019

@author: Daniela
"""
from readimg import read_img
import cv2
from scipy.spatial import distance
import pylab as plt 
import statistics

def euclidiana(img1,img2,a):
    histgramaNor = []
    histograma2Nor = []
    euclidianNor = []
    if a==0:
        for d in range(3):
            histNor = cv2.calcHist([img1], [d], None, [256], [1, 256])
            hist2Nor = cv2.calcHist([img2], [d], None, [256], [1, 256])
            histgramaNor.append(histNor)
            histograma2Nor.append(hist2Nor)
        
        for n in range(3):
            euclNor = distance.euclidean(histgramaNor[n], histograma2Nor[n])
            euclidianNor.append(euclNor)
        totaleucNor = statistics.mean(euclidianNor)
    else:
          histNor = cv2.calcHist([img1], [0], None, [256], [1, 256])
          hist2Nor = cv2.calcHist([img2], [0], None, [256], [1, 256])
          totaleucNor = distance.euclidean(histNor, hist2Nor)
   
    return totaleucNor
