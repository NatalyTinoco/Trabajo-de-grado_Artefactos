# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 00:27:48 2019

@author: Usuario
"""

import cv2
#import pylab as plt 
#from matplotlib import pyplot as plt
import numpy as np 
import glob 
import pandas as pd
#from scipy import stats
from contrasbrillo import contraste, brillo

brilloop2= []
brillomedia = []
contras = []


for image in glob.glob('*.jpg'):
    #image = '00002.jpg'
    im = cv2.imread(image)
    R,G,B=cv2.split(im)
    w,h,ch=im.shape
    brillon=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(w*h)
    brillomedia.append(np.mean(brillon))
    img=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
   
    contras.append(contraste(img))
    brilloop2.append(brillo(img))

datos = {'Brillo Opción 1':brillomedia,
         #'Brillo Opción 2':brilloop2,
         'Contraste':contras}

datos = pd.DataFrame(datos)
datos.to_excel('Características_brilloNO.xlsx')
    #plt.imshow(brillo,'Greys')
    #plt.show()
    #print(brillo)