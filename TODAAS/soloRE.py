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
from rOI import ROI

#tamañoA = []
#tamañoB = []

for image in glob.glob('*.jpg'):    
    im = cv2.imread(image)
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
                o=im[int(y1):int(y2),int(x1):int(x2)]
                dire='./DM/'+str(dm)+image
                cv2.imwrite(dire,o)
            if cls==0:
                re=re+1
                
#    if dm > 0:
       
        