# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:31:20 2019

@author: Nataly
"""

from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##

import glob
import cv2
import numpy as np 
from rOI import ROI
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

def contraststretching(img):
    #contrast Stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]

for imgfile in glob.glob("*.jpg"):
        
    img=cv2.imread(imgfile)   
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#    img=contraststretching(img)
    img=adaptativeequalization(img)
#    
    ima='./segROI/#5/Z3/'+imgfile
    imaROI=cv2.imread(ima,0)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
      
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    
    
    #img=tloga(img)
    R,G,B=cv2.split(img)
    
    #f,c,ch=img.shape
    #YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    #Y,U,V=cv2.split(YUV)      
    G=G*imaROI 
#    cv2.imshow('image',G)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
    V=G[y3:y3+h3,x3:x3+w3]
    
    Binary=V.copy()
    
    hist = cv2.calcHist([V],[0],None,[256],[0,255])
    hist=np.asarray(hist).astype(np.int)
    zz=list(range(0,len(hist)))
    for ii in range(len(hist)):
        zz[ii]=int(hist[ii])
    
    #hist=np.transpose(hist) 
#    hist=hist.tolist() 
    plt.plot(hist)
    
    plt.show()
    #peaks,_ = find_peaks(zz)
    #plt.plot(peaks, hist[peaks], "x")
    gradiente=np.gradient(zz[200:])
#    gradiente=np.gradient(zz)
#    plt.plot(gradiente)
#    plt.show()
    
    uu=find_nearest(gradiente,0)
    #maxcam=np.min(gradiente)
    gradiente=gradiente.tolist()
    umbral1 = gradiente.index(uu)
    umbral=200+umbral1
    #umbral=umbral1
    print(umbral)
    ta1,ta2=V.shape
    Binary=V.copy()
    for ff in range(ta1):
           for cc in range (ta2):
               if V[ff,cc]<umbral:
                   #if s[f,c]<h[f,c]:
                   Binary[ff,cc]=0
               else:
                   Binary[ff,cc]=255
    
    fila,Col=G.shape
    Binaryfinal=np.zeros((fila,Col)).astype(np.uint8)
    Binaryfinal[y3:y3+h3,x3:x3+w3]=Binary   
    
#    cv2.imshow('image',V)
#    cv2.waitKey(0)
#    cv2.imshow('image',Binaryfinal)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilatacion = cv2.dilate( Binaryfinal,kernel,iterations = 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close=cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, kernel)
    
    filetxt=imgfile[0:len(imgfile)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, img.shape)  
    re=0
    dm=0
#    cv2.imshow('image',close)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            dm=dm+1
        if cls==0:
            re=re+1
            
    if dm>0 and re==0:
        dire='./segmentacionSthele_CanalG_ConEcuAdaptativa/DM/'+imgfile
        cv2.imwrite(dire,close)
        print('dm')
    else:
        direM='./segmentacionSthele_CanalG_ConEcuAdaptativa/RE/'+imgfile
        cv2.imwrite(direM,close)
        print('re')


    
    

