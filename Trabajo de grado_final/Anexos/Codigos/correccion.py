# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:10:34 2019

@author: Daniela y Nataly 

Descripción del código: Código donde se hace uso de la función de open cv de inapiting con un radio de 10.
"""
import cv2

def inpaintingTA(ima,mask):
    telea = cv2.inpaint(ima,mask, 10, cv2.INPAINT_TELEA)
    return telea