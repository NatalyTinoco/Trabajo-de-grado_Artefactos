# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:52:55 2019

@author: Nataly
"""

#from readimg import read_img
import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt
from rOI import ROI
from skimage.feature import greycomatrix, greycoprops
import skimage.feature


#tamañoA = []
#tamañoB = []
def Fourier(inA):
    f = np.fft.fft2(inA)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    return fourier 
    
def GLCM (imA):
        a=int(np.max(imA))
        g = skimage.feature.greycomatrix(imA, [1], [0], levels=a+1, symmetric=False, normed=True)                  
        contraste=skimage.feature.greycoprops(g, 'contrast')[0][0]
        energia=skimage.feature.greycoprops(g, 'energy')[0][0]
        homogeneidad=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
        correlacion=skimage.feature.greycoprops(g, 'correlation')[0][0]
        disimi= greycoprops(g, 'dissimilarity') 
        ASM= greycoprops(g, 'ASM')
        entropia=skimage.measure.shannon_entropy(g) 
        return contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia
#                    plt.imshow(cropped)
    
contrast=[]
energi=[]
homogenei=[]
correlaci=[]
disi=[]
AS=[]
entrop=[]

contrastRE=[]
energiRE=[]
homogeneiRE=[]
correlaciRE=[]
disiRE=[]
ASRE=[]
entropRE=[]

contrastNO=[]
energiNO=[]
homogeneiNO=[]
correlaciNO=[]
disiNO=[]
ASNO=[]
entropNO=[]


for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        
    for z in range(c):
        im[:,:,z]=im[:,:,z]*imaROI
    
    filetxt=image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    dm=0
    imunda=0
    imSinBBOX=im.copy()
    tamañoA = 100
    tamañoB = 100
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            dm=dm+1
            
        if cls==0:
            re=re+1        
            print(re)
        if cls==2:
            imunda=imunda+1

    if imunda > 0 and dm==0 and re==0:
        dire='./imaduda/art2/' +image 
        cv2.imwrite(dire,im)
    if re==0 and dm==0 and imunda==0:
        dire='./imaduda/otroart/' +image 
        cv2.imwrite(dire,im)
    
            
            
            
            
        
           