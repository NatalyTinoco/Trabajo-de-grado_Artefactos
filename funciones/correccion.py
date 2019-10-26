# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:10:34 2019

@author: Usuario
"""
from Filtros import filtromediana
from skimage.restoration import inpaint
import numpy as np
import cv2

def suavizado(imgOri,imSeg,nmask):
    img=imgOri.copy()
    nu=nmask
    img_suav= filtromediana(img,nu)
    img_suav2=img_suav.copy()
    img_seg=cv2.normalize(imSeg, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    for z in range(3):
        img_suav2[:,:,z]=img_suav[:,:,z]*img_seg
    
    img2=imgOri.copy()
    img3=img2.copy()
    
    for z in range(3):
        img2[:,:,z]=img3[:,:,z]*(1-img_seg)
        
    img_result=img2+img_suav2  
    
    return img_result

def inpaintingB(imgOri,imSeg):
    image_defect = imgOri.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(imSeg)] = 0

    image_result =inpaint.inpaint_biharmonic(imgOri, imSeg,multichannel=True)
    
    image_result =inpaint.inpaint_biharmonic(image_defect, imSeg, multichannel=3)
    
    imgs= cv2.normalize(image_result, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    return imgs

def inpaintingNS(ima,mask):
#    ns = cv2.inpaint(ima,mask, 3, cv2.INPAINT_NS)
    ns = cv2.inpaint(ima,mask, 10, cv2.INPAINT_NS)
#    ns = cv2.inpaint(ima,mask, 15, cv2.INPAINT_NS)
#    ns = cv2.inpaint(ima,mask, 1, cv2.INPAINT_NS)
    
    return ns

def inpaintingTA(ima,mask):
#    telea = cv2.inpaint(ima,mask, 3, cv2.INPAINT_TELEA)
    telea = cv2.inpaint(ima,mask, 10, cv2.INPAINT_TELEA)
#    telea = cv2.inpaint(ima,mask, 15, cv2.INPAINT_TELEA)
#    telea = cv2.inpaint(ima,mask, 1, cv2.INPAINT_TELEA)
    return telea