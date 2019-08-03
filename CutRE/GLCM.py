# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 00:15:03 2019

@author: Nataly
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from skimage import data
def tloga(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img

def normalizacionMaxMin(img):
    img2 = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img2

import glob

contraste=np.zeros((4075,1))
energia=np.zeros((4075,1))
homogeneidad=np.zeros((4075,1))
correlacion=np.zeros((4075,1))
disimi=np.zeros((4075,1))
ASM=np.zeros((4075,1))
entropia=np.zeros((4075,1))
i=0               
for file in glob.glob("*.jpg"): 
    img=cv2.imread(file)
    img=normalizacionMaxMin(img)
    #plt.imshow(img) 
    #plt.show()
    YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y,U,V=cv2.split(YUV)
    f = np.fft.fft2(V)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    #plt.imshow(fourier, cmap = 'gray')
    #plt.show()
    a=int(np.max(fourier))
    g = skimage.feature.greycomatrix(fourier, [1], [0], levels=a+1, symmetric=False, normed=True) 
   
    contraste[i,0]=skimage.feature.greycoprops(g, 'contrast')[0][0]
    energia[i,0]=skimage.feature.greycoprops(g, 'energy')[0][0]
    homogeneidad[i,0]=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
    correlacion[i,0]=skimage.feature.greycoprops(g, 'correlation')[0][0]
    disimi[i,0]= greycoprops(g, 'dissimilarity') 
    ASM[i,0]= greycoprops(g, 'ASM')
    entropia[i,0]=skimage.measure.shannon_entropy(g)
    i=i+1
    
import openpyxl

doc = openpyxl.load_workbook('GLCM.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
#table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
#table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ','BA','BB','BC','BD','BE','BF','BG','BH']
table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI']

#table = ['A','B','C']

i=0
ii=0
#for a in range (int(len(table)/4)):
a=0
    #print(i)
    #print(a)
for x in range ((len(contraste[:,0]))):
    hoja[table[1]+ str (x+4)]=contraste[x,a]
    hoja[table[3]+ str (x+4)]=energia[x,a]
    hoja[table[5]+ str (x+4)]=homogeneidad[x,a]
    hoja[table[7]+ str (x+4)]=correlacion[x,a]
    hoja[table[9]+ str (x+4)]=disimi[x,a]
    hoja[table[11]+ str (x+4)]=ASM[x,a]
    hoja[table[13]+ str (x+4)]=entropia[x,a]
    
#print(table[i])
#print(table[i+1])
#print(table[i+2])
print(x,a)
i=(a+1)*4
#print(a)
doc.save("GLCM.xlsx")