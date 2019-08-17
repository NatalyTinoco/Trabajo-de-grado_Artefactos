# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 22:44:45 2019

@author: Nataly
"""
import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt

#tamañoA = []
#tamañoB = []

for image in glob.glob('*.jpg'):    
#    image='00317.jpg'
    im = cv2.imread(image)
    filetxt=image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    dm=0
    ta=im.shape
    ta=list(ta)
    mascara=np.zeros([ta[0],ta[1]])
    for b in boxes_abs:
            cls, x1, y1, x2, y2 = b
            if cls == 3:
                dm=dm+1
                artefacto=im[int(y1):int(y2),int(x1):int(x2),:]
                tavta=artefacto.shape
                tavta=list(tavta)
                factor=0                  
                for y in range(int(y1),int(y2)):
                    for x in range(int(x1),int(x2)):
                        mascara[y,x]=1
#             
#                plt.imshow(mascara,'Greys')
#                plt.show()
                print(dm,y1,y2,x1,x2)
            if cls==0:
                re=re+1
                
    mascara=mascara.astype(np.uint8)
    mascara = cv2.normalize(mascara, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    dire='./SegmentacionGT/'+image
    cv2.imwrite(dire,mascara)   
#       
#    plt.imshow(mascara,'Greys')
#    plt.show()
#           
#    if dm > 0:
       
        