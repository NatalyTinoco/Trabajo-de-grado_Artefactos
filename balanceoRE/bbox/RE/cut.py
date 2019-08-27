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

from skimage import exposure
import pylab as plt 
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    #Adaptative Equalization
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)

    imha=cv2.merge((imhaR,imhaG,imhaB))
    
    return imha
for image in glob.glob('*.jpg'):    
    image='00377.-3.jpg'
    im = cv2.imread(image)
    im = cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    im=adaptativeequalization(im)
    im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    R,G,B=cv2.split(im)    
    original=G.copy()
#    original=cv2.resize(original,(600,500))
    plt.imshow(im)
    plt.show()
    
    plt.hist(G.ravel(),256,[0,256])
    plt.show()
    
#    filetxt=image[0:len(image)-3]+'txt'
##    f=open(datafolder/filetxt)
##          
##    bboxfile=f.read()
#    boxes = read_boxes(filetxt)
#    
#    boxes_abs = yolo2voc(boxes, im.shape)  
#    re=0
#    for b in boxes_abs:
#            cls, x1, y1, x2, y2 = b
#            if cls == 0:
#                re=re+1
#                cropped=im[int(y1):int(y2),int(x1):int(x2),:]
##                plt.imshow(cropped)
##                plt.show()
#                dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/RE/1/'+image[0:len(image)-3]+'-'+str(re)+'.jpg'
#                cv2.imwrite(dire,cropped)
#                
#                
#                
                
                
                
                
                
                
                
                
                