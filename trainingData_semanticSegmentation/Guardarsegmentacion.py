# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 23:56:42 2019

@author: Nataly
"""
import sys
path = "C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/funciones"
sys.path.insert(1, path)


#from readimg import read_img #leer imagines ### img=read_img(imgfile)##
import cv2
import xlrd
#from scipy import stats # importando scipy.stats
#   import numpy as np
import tifffile as tiff

workbook = xlrd.open_workbook("segentadasSI.xlsx")

sheet_names = workbook.sheet_names()

xl_sheet = workbook.sheet_by_name(sheet_names[0])

direSave = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/imagenesSeg/si/'

for col in range(xl_sheet.ncols):
    ttfile = xl_sheet.cell_value(1, col)
    if ttfile != '':
        ttfile=ttfile+'tif'
        tif = tiff.imread(ttfile)
        
        direIn=direSave+'Instrumento/'+ttfile
        direRe=direSave+'RE/'+ttfile
        direSat=direSave+'Saturacion/'+ttfile
        cv2.imwrite(direIn,tif[0])
        cv2.imwrite(direRe,tif[1])
        cv2.imwrite(direSat,tif[4])
    
#    print(imgfile)

#%%
import glob
import os

for image in glob.glob("*mask.tif"):
    imageNew = image.split("_mask")
    dst = imageNew[0]+'.tif'
    
    os.rename(image, dst)
    print(image)
    
    
#%%
import shutil

src = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/trainingData_semanticSegmentation/0_original_images/'
dest = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/'
for col in range(xl_sheet.ncols):
    ttfile = xl_sheet.cell_value(1, col)
    if ttfile != '':
        ttfile=ttfile+'jpg'
        shutil.copyfile(src+ttfile,dest+ttfile)
