# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:08:21 2019

@author: Nataly
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pylab as plt 
from matplotlib import pyplot as plt
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr    
import xlrd
from skimage.filters.rank import minimum
from skimage.morphology import disk

def log(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img
    
def filtrominimo(img):
    B,G,R=cv2.split(img)
    fR = minimum(R, disk(3))
    fG = minimum(G, disk(3))
    fB = minimum(B, disk(3))
    imh=img
    imh[:,:,0]=fB
    imh[:,:,1]=fG
    imh[:,:,2]=fR
    return imh

workbook = xlrd.open_workbook("ListasubRE.xlsx")

sheet = workbook.sheet_by_index(0)

for col in range(sheet.ncols):
    imgfile = sheet.cell_value(0, col)  
    print(imgfile)
    img=cv2.imread(imgfile+'jpg')
    img=log(img)
    #img=filtrominimo(img)
     
    ima='./subNormRE/segROI/'+imgfile+'jpg'
    imaROI=cv2.imread(ima,0)
    #imaROI=imaROI[:,:,2]
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)   
    #YUV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y,U,V=cv2.split(YUV)    
    Y = cv2.normalize(Y, None, 0, 360, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    ima=Y*imaROI
    cv2.imshow('image',U)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
#plt.boxplot(bank.balance)
    ima=ima.reshape(-1)
    plt.ion()
    plt.boxplot([ima], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
    plt.xticks([1], ['img'], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
    plt.ylabel('Img')
    plt.show()
