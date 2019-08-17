# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:11:56 2019

@author: Nataly
"""

import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt
from rOI import ROI
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from scipy.stats import kurtosis
import statistics as stats
import pywt
import pywt.data



dm=0
re=0
no=0
    
for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    #cv2.imshow('Grays',imaROI)
    #cv2.destroyAllWindows()
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
    V=V*imaROI
        
    for z in range(c):
        im[:,:,z]=im[:,:,z]*imaROI
    
    
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    filetxt=image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    imunda=0
    imSinBBOX=im.copy()
    tamañoA = 500
    tamañoB = 500       
    ree=0
    dmm=0
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            print('DM')
            dmm=dmm+1
            #print(image,dm)
            a,b= V[int(y1):int(y2),int(x1):int(x2)].shape
            #tamañoA,tamañoB=tama(a,b)
            V1= V[int(y1):int(y2),int(x1):int(x2)]
            imdm= im[int(y1):int(y2),int(x1):int(x2)]
            vecesA = int(a/tamañoA)
            vecesB = int(b/tamañoB)
            
            for f in range(0,a-tamañoA,tamañoA):
                for c in range(0,b-tamañoB,tamañoB):
                    #print(f,c)
                    cropped = V1[f:f+tamañoA,c:c+tamañoB]
                    croppedrgb = imdm[f:f+tamañoA,c:c+tamañoB]
                   
                    #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
                    if c==tamañoB*vecesB-tamañoB:
                        cropped = V1[f:f+tamañoA,c:]
                        croppedrgb = imdm[f:f+tamañoA,c:]
                        #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
                    if f==tamañoA*vecesA-tamañoA:
                         #print('ola')
                         if c==tamañoB*vecesB-tamañoB:
                            cropped = V1[f:,c:]
                            croppedrgb = im[f:,c:]
                       
                             #test2[f:,c:]=test[f:,c:]
                         else:
                             cropped = V1[f:,c:c+tamañoB]
                             croppedrgb = imdm[f:,c:c+tamañoB]
                       
                             #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                             #print('dani')
                    dm=dm+1  
                    cropped=cv2.resize(cropped,(500,500))      
                    croppedrgb=cv2.resize(croppedrgb,(500,500))  
                    diredm='./500x500/DM/color/'+str(dm)+image
                    cv2.imwrite(diredm,croppedrgb)
                    diredm2='./500x500/DM/V/'+str(dm)+image
                    cv2.imwrite(diredm2,cropped)
                    #print('DM',dm)
#                    plt.imshow(croppedrgb)
#                    plt.show()
       
                   

        if cls==0:
            ree=ree+1        
            #print(re)
        if cls==2:
            imunda=imunda+1
    
    if ree > 0 and dmm==0 and imunda==0:
            
            inta=V[y3:y3+h3,x3:x3+w3]
            imre=im[y3:y3+h3,x3:x3+w3]
            aa,bb=inta.shape
#            tamañoA,tamañoB=tama(aa,bb)
            vecesA = int(aa/tamañoA)
            vecesB = int(bb/tamañoB)
        
            for f in range(0,aa-tamañoA,tamañoA):
                for c in range(0,bb-tamañoB,tamañoB):
                    cropped2 = inta[f:f+tamañoA,c:c+tamañoB]
                    croppedrgb2 = imre[f:f+tamañoA,c:c+tamañoB]
                    if c==tamañoB*vecesB-tamañoB:
                        cropped2 = inta[f:f+tamañoA,c:]
                        croppedrgb2 = imre[f:f+tamañoA,c:]
                    if f==tamañoA*vecesA-tamañoA:
                         if c==tamañoB*vecesB-tamañoB:
                            cropped2 = inta[f:,c:]
                            croppedrgb2 = imre[f:,c:]
                         else:
                             cropped2 = inta[f:,c:c+tamañoB]
                             croppedrgb2 = imre[f:,c:c+tamañoB]
                    cropped2=cv2.resize(cropped2,(500,500))      
                    croppedrgb2=cv2.resize(croppedrgb2,(500,500))
                    re=re+1
                    direre='./500x500/RE/color/'+str(re)+image
                    cv2.imwrite(direre,croppedrgb2)
                    direre2='./500x500/RE/V/'+str(re)+image
                    cv2.imwrite(direre2,cropped2)
                    #print('RE',re)
#                    plt.imshow(croppedrgb2)
#                    plt.show()

            
    if ree==0 and dmm==0 and imunda==0:
            inta3=V[y3:y3+h3,x3:x3+w3]
            aaa,bbb=inta3.shape
            imno=im[y3:y3+h3,x3:x3+w3]
           
#            tamañoA,tamañoB=tama(aaa,bbb)
            vecesA = int(aaa/tamañoA)
            vecesB = int(bbb/tamañoB)
        
            for f in range(0,aaa-tamañoA,tamañoA):
                for c in range(0,bbb-tamañoB,tamañoB):
                    cropped3 = inta3[f:f+tamañoA,c:c+tamañoB]
                    croppedrgb3 = imno[f:f+tamañoA,c:c+tamañoB]
                    if c==tamañoB*vecesB-tamañoB:
                        cropped3 = inta3[f:f+tamañoA,c:]
                        croppedrgb3 = imno[f:f+tamañoA,c:]
                    if f==tamañoA*vecesA-tamañoA:
                         if c==tamañoB*vecesB-tamañoB:
                            cropped3 = inta3[f:,c:]
                            croppedrgb3 = imno[f:,c:]
                         else:
                             cropped3 = inta3[f:,c:c+tamañoB]
                             croppedrgb3 = imno[f:,c:c+tamañoB]
                    cropped3=cv2.resize(cropped3,(500,500))      
                    croppedrgb3=cv2.resize(croppedrgb3,(500,500)) 
                    no=1+no
                    direno='./500x500/NO/color/'+str(no)+image
                    cv2.imwrite(direno,croppedrgb3)
                    direno2='./500x500/NO/V/'+str(no)+image
                    cv2.imwrite(direno2,cropped3)
#                    plt.imshow(croppedrgb3)
#                    plt.show()
    print(image,dmm,ree,imunda)