# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 23:56:42 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from Normalizacion import  normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,tloga
import cv2
import pylab as plt
from scipy import stats # importando scipy.stats
import numpy as np
import tifffile as tiff
import glob

for file in glob.glob("*.jpg"):
    imgfile=file
    imgfile=imgfile
    img = read_img(imgfile)
    ima=normalizacionMaxMin(img)
    dire='./subNormDM/'+imgfile[0:len(imgfile)-4]+'.jpg'
    
    cv2.imwrite(dire,ima)
    k = cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print(imgfile)
