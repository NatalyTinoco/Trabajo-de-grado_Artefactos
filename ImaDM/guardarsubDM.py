# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:22:16 2019

@author: Nataly
"""


from readimg import read_img #leer imagines ### img=read_img(imgfile)##
import cv2
import pylab as plt
import xlrd

workbook = xlrd.open_workbook("ListasubDM.xlsx")

sheet = workbook.sheet_by_index(0)

for col in range(sheet.ncols):
    imgfile = sheet.cell_value(0, col)
    imgfile=imgfile+'jpg'
    print(imgfile)
    img = read_img(imgfile)
    dire='./subDM/'+imgfile[0:len(imgfile)-4]+'.jpg'
    
    cv2.imwrite(dire,img)
    k = cv2.waitKey(1000)
    cv2.destroyAllWindows()
    