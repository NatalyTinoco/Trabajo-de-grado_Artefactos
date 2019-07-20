# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:43:23 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio

import cv2
import pylab as plt
import xlrd
from scipy import stats # importando scipy.stats
import numpy as np
import tifffile as tiff
import glob

for imgfile in glob.glob("*.jpg"):
    img = read_img(imgfile)
    ima=filtrominimo(img)
    dire='./FiltroDM/'+imgfile[0:len(imgfile)-4]+'.jpg'
    
    cv2.imwrite(dire,ima)
    k = cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print(imgfile)