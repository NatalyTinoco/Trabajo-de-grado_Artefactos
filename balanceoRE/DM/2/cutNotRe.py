# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:36:19 2019

@author: Usuario
"""
from pathlib import Path
datafolder=Path("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/txt")
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS')

#from readimg import read_img
import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt
from rOI import ROI

#tamañoA = []
#tamañoB = []

for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
#    aa,bb,c = im.shape    
#    imaROI=ROI(im)
#    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#   
#    #cv2.imshow('Grays',imaROI)
#    #cv2.destroyAllWindows()
#    for z in range(c):
#        im[:,:,z]=im[:,:,z]*imaROI
#        
#    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#    areas = [cv2.contourArea(c) for c in contours]
#    max_index = np.argmax(areas)
#    cnt=contours[max_index]
#    x,y,w,h = cv2.boundingRect(cnt)
   
#    a,b,ch = im[y:y+h,x:x+w].shape
    a,b,ch = im.shape
    tamañoA = 48
    tamañoB = 46
    vecesA = int(a/tamañoA)
    vecesB = int(b/tamañoB)
    re=0
    for f in range(0,a-tamañoA,tamañoA):
        for c in range(0,b-tamañoB,tamañoB):
            cropped = im[f:f+tamañoA,c:c+tamañoB]
           
            #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
            if c==tamañoB*vecesB-tamañoB:
                cropped = im[f:f+tamañoA,c:]
           
                #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
            if f==tamañoA*vecesA-tamañoA:
                 #print('ola')
                 if c==tamañoB*vecesB-tamañoB:
                    cropped = im[f:,c:]
           
                     #test2[f:,c:]=test[f:,c:]
                 else:
                     cropped = im[f:,c:c+tamañoB]
                     #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                     #print('dani')
#            plt.imshow(cropped)
#            plt.show() 
            re=re+1
            dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/DM/2/'+image[0:len(image)-3]+'-'+str(re)+'.jpg'
            cv2.imwrite(dire,cropped)
            # 
            
            #print(test[f:f+tamañoA,c:c+tamañoB])
        

#promedioa = np.mean(tamañoA)
#promediob = np.mean(tamañoB)
