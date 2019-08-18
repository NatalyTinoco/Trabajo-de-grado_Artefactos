# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:20:34 2019

@author: Nataly
"""
from pathlib import Path
datafolder=Path("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/txt")
import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/TODAAS')
import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
#from readboxes import read_boxes
from matplotlib import pyplot as plt

def read_boxes(txtfile):
    import numpy as np
    lines = []
    
    with open(datafolder/filetxt) as f:
    
        for line in f:
            line = line.strip()
            box = np.hstack(line.split()).astype(np.float)
            box[0] = int(box[0])
            lines.append(box)
    return np.array(lines)

for image in glob.glob('*.jpg'):    
    im = cv2.imread(image)
    filetxt=image[0:len(image)-3]+'txt'
#    f=open(datafolder/filetxt)
#          
#    bboxfile=f.read()
    boxes = read_boxes(filetxt)
    
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    for b in boxes_abs:
            cls, x1, y1, x2, y2 = b
            if cls == 0:
                re=re+1
                cropped=im[int(y1):int(y2),int(x1):int(x2),:]
#                plt.imshow(cropped)
#                plt.show()
                dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/RE/9/'+image[0:len(image)-3]+'-'+str(re)+'.jpg'
                cv2.imwrite(dire,cropped)
                
                
                
                
                
                
                
                
                
                
                
                