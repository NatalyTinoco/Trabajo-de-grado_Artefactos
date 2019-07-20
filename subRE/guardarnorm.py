# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 23:56:42 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from Normalización import  normalizacionMaxMin,normalizacionrgb,normalizacionintensi,estandarizacion,log
import cv2
import pylab as plt
import xlrd
from scipy import stats # importando scipy.stats
import numpy as np
import tifffile as tiff

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

for col in range(sheet.ncols):
    imgfile = sheet.cell_value(0, col)
    imgfile=imgfile+'jpg'
    img = read_img(imgfile)
    ima=log(img)
    dire='./subNormRE/'+imgfile[0:len(imgfile)-4]+'.jpg'
    
    cv2.imwrite(dire,ima)
    k = cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print(imgfile)
