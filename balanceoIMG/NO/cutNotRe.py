# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:36:19 2019

@author: Usuario
"""

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
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    #cv2.imshow('Grays',imaROI)
    #cv2.destroyAllWindows()
    for z in range(c):
        im[:,:,z]=im[:,:,z]*imaROI
        
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    #""" 
#    cv2.imshow("Show",im[y:y+h,x:x+w])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    imf=im.copy()
#    cv2.rectangle(imf,(x,y),(x+w,y+h),(0,255,0),2)
#    cv2.imshow("Show",imf)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    #"""
    #plt.imshow(im)
    #plt.show()
    #imagenROI=im*imaROI
#    filetxt=image[0:len(image)-3]+'txt'      
#    bboxfile=filetxt
#    boxes = read_boxes(bboxfile)
#    boxes_abs = yolo2voc(boxes, im.shape)  
#    re=0
#    dm=0
#    
#    for b in boxes_abs:
#        cls, x1, y1, x2, y2 = b
#        if cls == 3:
#            dm=dm+1
#        if cls==0:
#            re=re+1
#    if dm > 0 and re == 0:
#        
        #test =  np.random.random_sample((19,17))
        #test2 =  np.zeros((19,17))
   
    a,b,ch = im[y:y+h,x:x+w].shape
    tamañoA = 48
    tamañoB = 46
    vecesA = int(a/tamañoA)
    vecesB = int(b/tamañoB)

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
            plt.imshow(cropped)
            plt.show() 
            dire='./cutNotRe_siDM/'+str(c)+str(f)+'-' +image 
            cv2.imwrite(dire,cropped)
            # 
            
            #print(test[f:f+tamañoA,c:c+tamañoB])
        

#promedioa = np.mean(tamañoA)
#promediob = np.mean(tamañoB)
