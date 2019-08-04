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
    image = '00002.jpg'
    im = cv2.imread(image)
    a,b,c = im.shape    
    imaROI=ROI(im)
    plt.imshow(imaROI,'Grays')
    plt.show()
    filetxt=image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    dm=0
    
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            dm=dm+1
        if cls==0:
            re=re+1
    if dm > 0 and re == 0:
        
        #test =  np.random.random_sample((19,17))
        #test2 =  np.zeros((19,17))
       
        #a,b = test.shape
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
                dire='./cutNotRe_siDM/'+ image 
                cv2.imwrite(dire,cropped)
                # 
                
                #print(test[f:f+tamañoA,c:c+tamañoB])
            

#promedioa = np.mean(tamañoA)
#promediob = np.mean(tamañoB)
