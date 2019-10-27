# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:35:27 2019

@author: Daniela y Nataly 

Descripción del código: Código donde se realiza la ecualización adaptativa de ventanas de 8x8 

"""

import cv2
def adaptativeequalization(img):  
    imR, imG, imB=cv2.split(img) 
    clahe=cv2.createCLAHE(2,(8,8))
    imhaR=clahe.apply(imR)
    imhaG=clahe.apply(imG)
    imhaB=clahe.apply(imB)
    imha=cv2.merge((imhaR,imhaG,imhaB))
    return imha
