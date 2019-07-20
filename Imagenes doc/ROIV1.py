# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:33:51 2019

"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
from plotboxes import plot_boxes #graficar bbox ## plot_boxes(ax, boxes)##
from readobjnames import read_obj_names  ## classes = read_obj_names(classfile) ##

import cv2
import glob
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
from pylab import *
import xlrd

from PIL import ImageEnhance
from PIL import Image 

from skimage.filters import threshold_otsu
from skimage.filters import threshold_li
from skimage.filters import threshold_minimum
from skimage.filters import try_all_threshold

workbook = xlrd.open_workbook("imagenesQueNosSirven.xlsx")
sheet = workbook.sheet_by_index(0)

c=0
a=1
i=0

for col in range(sheet.ncols):
    filetxt = sheet.cell_value(0, col)
    for col2 in range(sheet.ncols):
        file = sheet.cell_value(1, col)    
#for filetxt in glob.glob("*.txt"):
  #  for file in glob.glob("*.jpg"):
        c=c+1
        if c==a:
            f=file
            c=c+1  
            #print(f)
            break
    imgfile=f
    ima=read_img(imgfile)
    imB,imG,imR=cv2.split(ima)
    imA=cv2.cvtColor(ima,cv2.COLOR_RGB2YUV)
    Y,U,II=cv2.split(imA)
    #II=imR
    #fig, ax = try_all_threshold(II, figsize=(10, 8), verbose=False)
    #plt.show()
    #thresh = threshold_otsu(II)
    thresh = threshold_li(II)    
    #thresh=threshold_minimum(II)
    binary = II > thresh
    binary=binary/255    
    binary = (binary*255).astype(np.uint8)
    
    #"""### Transformaciones Morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close=cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    
    # region de interes 
    _,contours,_ = cv2.findContours(close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    #img=np.asarray(ima)
    cv2.rectangle(ima,(x,y),(x+w,y+h),(0,255,0),2)
    dire='./ROI/ROI_V1(Suma RGB)/'+f[0:len(f)-4]+'.jpg'
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   
    cv2.imwrite(dire,ima)
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
    #
    print(f)
    #print(filetxt)
    a=a+1
    c=0

