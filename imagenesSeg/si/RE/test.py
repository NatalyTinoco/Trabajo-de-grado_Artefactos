# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 00:52:50 2019

@author: Usuario
"""

import sys
sys.path.insert(1,'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/funciones')

from skimage.measure import compare_ssim as ssim
import cv2
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sthele2006 import sthele2006
from ventanIDEA import ventanIDEA
from umbralAdapArt import umbralAdapArt


def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    Dice=2. * intersection.sum() / (im1.sum() + im2.sum())
    return Dice    


image = '00000.tif'
original = cv2.imread(image,0)
image2 = image.split(".")
imageFile = image2[0]+'.jpg'
tecnica1, y, v= umbralAdapArt(imageFile) 



cv2.imshow('hsv',yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()