# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 00:27:48 2019

@author: Usuario
"""

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
import glob 
import pandas as pd
from scipy import stats

brillomoda = []
brillomedia = []

for image in glob.glob('*.jpg'):
    #image = '00002.jpg'
    im = cv2.imread(image)
    R,G,B=cv2.split(im)
    w,h,ch=im.shape
    brillo=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(w*h)
    brillomedia.append(np.mean(brillo))
    brillomoda.append(stats.mode(brillo))

datos = {'Media':brillomedia}

datos = pd.DataFrame(datos)
datos.to_excel('Características_brilloDM.xlsx')
    #plt.imshow(brillo,'Greys')
    #plt.show()
    #print(brillo)