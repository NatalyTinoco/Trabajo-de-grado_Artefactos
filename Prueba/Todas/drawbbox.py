# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:08:02 2019

@author: Nataly
"""


import sys
sys.path.insert(1, 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from yolovoc import yolo2voc
from readboxes import read_boxes
from test_todoRE import test_all_RE
from test_todoDM import test_all_DM
import numpy as np
import cv2 
import glob
import pylab as plt 
from matplotlib import pyplot as plt
from pylab import *

def draw_boxes(boxes,im,r):
    for b in boxes:
        cls, x1, y1, x2, y2 = b
        print('clase',cls)
        if cls == 0 or cls == 3:
            if cls==3:
                cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
            if cls==0:
                cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)  
                
    r=r+1
    return r,im


for file in glob.glob("*.jpg"): 
    im=cv2.imread(file)
    print(file)
    filetxt=file[0:len(file)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes = yolo2voc(boxes, im.shape)  
    for b in boxes:
        cls, x1, y1, x2, y2 = b
       
        if cls == 0 or cls == 3:
            if cls==3:
                cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
                print('clase',cls,x1,x2,y1,y2)
            if cls==0:
                cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)  
                print('clase',cls,x1,x2,y1,y2)
                
#    cv2.imshow('',im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    dire='./draw/'+file
    cv2.imwrite(dire, im)

