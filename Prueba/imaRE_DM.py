"""
Created on Fri Aug 16 22:44:45 2019

@author: Nataly
"""
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
ambas = []
for image in glob.glob('*.jpg'):    
#    image='00317.jpg'
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
            if cls==0:
                re=re+1
    if re > 0 and dm > 0:
        dire1='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/DM&RE/'+image
        cv2.imwrite(dire1,im)   
    if re>0 and dm==0:
        dire2='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/RE/'+image
        cv2.imwrite(dire2,im) 
    if re==0 and dm>0:
        dire3='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/DM/'+image
        cv2.imwrite(dire3,im)   
    if re==0 & dm==0:
        dire4='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/NO/'+image
        cv2.imwrite(dire4,im)   
