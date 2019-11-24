# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:55:18 2019

@author: Daniela y Nataly 

Descripción del código: Código donde se realiza la normalización Min-max para obtener un nuevo rango de intensidad de 0 a 255.

"""
import cv2
def normalizacionMaxMin(img):
    img2 = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img2

