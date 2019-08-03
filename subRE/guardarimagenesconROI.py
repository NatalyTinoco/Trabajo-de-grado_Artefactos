# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:07:22 2019

@author: Nataly
"""
import cv2
import xlrd
workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)
for col in range(sheet.ncols):
    imgfile = sheet.cell_value(0, col)  
    imgf='./subNormRE/FiltroRE/Espaciodecolor/V/'+imgfile+'jpg'
    img='./subNormRE/segROI/'+imgfile+'jpg'
    print(img)
    ima=cv2.imread(imgf,0)
    imaROI=cv2.imread(img,0)
    #print(ima.shape,imaROI.shape)
    ima=ima*imaROI
    #for z in range(3):
    #    ima[:,:,z]=ima[:,:,z]*imaROI
    dire='./subNormRE/FiltroRE/Espaciodecolor/V/'+imgfile+'jpg'
    #img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   
    cv2.imwrite(dire,ima)