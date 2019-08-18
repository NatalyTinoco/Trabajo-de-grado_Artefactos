# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:36:19 2019

@author: Usuario
"""
from pathlib import Path
datafolder=Path("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/txt")
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from yolovoc import yolo2voc
from readboxes import read_boxes
from rOI import ROI
from ventaneo import ventaneoo


import cv2
import numpy as np
import glob

from matplotlib import pyplot as plt


   
for image in glob.glob('*.jpg'):
    print(image)
    # image = '00002.jpg'
    im = cv2.imread(image)
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    #cv2.imshow('Grays',imaROI)
    #cv2.destroyAllWindows()
    for z in range(c):
        im[:,:,z]=im[:,:,z]*imaROI
        
#    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#    areas = [cv2.contourArea(c) for c in contours]
#    max_index = np.argmax(areas)
#    cnt=contours[max_index]
#    x,y,w,h = cv2.boundingRect(cnt)
#    a,b,ch = im[y:y+h,x:x+w].shape
    a,b,ch = im.shape
#    print(a)
#    plt.imshow(im)
#    plt.show()
    print('50x50')
    tamañoA=50
    tamañoB=50
    re=0
#    if b<tamañoB and a<tamañoA:
#        cropped=im
#        dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/DM/1/'+image[0:len(image)-3]+'-'+str(re)+'.jpg'
#        cv2.imwrite(dire,cropped)
#    else:
    for f in range(0,a-tamañoA,tamañoA):
       for c in range(0,b-tamañoB,tamañoB):
            cropped=ventaneoo(tamañoA, tamañoB,a,b,f,c, im)
            dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/DM/3/'+image[0:len(image)-3]+'-'+str(re)+'.jpg'
            cv2.imwrite(dire,cropped)
            re=re+1
                
    print('150x150')
    tamañoaA=150
    tamañoaB=150
    rea=0
#    if b<tamañoaB and a<tamañoaA:
#        croppeda=im
#        dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/1/'+image[0:len(image)-3]+'-'+str(rea)+'.jpg'
#        cv2.imwrite(dire,croppeda)
#    else:
    for fa in range(0,a-tamañoaA,tamañoaA):
       for ca in range(0,b-tamañoaB,tamañoaB):
            croppeda=ventaneoo(tamañoaA, tamañoaB,a,b,fa,ca, im)
            dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/3/'+image[0:len(image)-3]+'-'+str(rea)+'.jpg'
            cv2.imwrite(dire,croppeda)
            rea=rea+1
    print('500x500')            
    tamañoa1A=500
    tamañoa1B=500
    rea1=0
#    if b<tamañoa1B and a<tamañoa1A:
#        croppeda1=im
#        dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/DM/1/'+image[0:len(image)-3]+'-'+str(rea1)+'.jpg'
#        cv2.imwrite(dire,croppeda1)
#    else:
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
       for ca1 in range(0,b-tamañoa1B,tamañoa1B):
            croppeda1=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, im)
            dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/DM/3/'+image[0:len(image)-3]+'-'+str(rea1)+'.jpg'
            cv2.imwrite(dire,croppeda1)
            rea=rea+1
#        print(image)
#   
#            dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/NO/2/'+image[0:len(image)-3]+'-'+str(re)+'.jpg'
#            cv2.imwrite(dire,cropped)
            # 
            
            #print(test[f:f+tamañoA,c:c+tamañoB])
        

#promedioa = np.mean(tamañoA)
#promediob = np.mean(tamañoB)
