# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:27:13 2019

@author: Nataly
"""
from skimage.measure import compare_ssim as ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

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
import glob

#similitud=np.zeros((307,16))
#mSSIM=np.zeros((307,16))
#mMSE=np.zeros((307,16))
#mDICE1=np.zeros((307,16))
#mDICE2=np.zeros((307,16))
mSSIM=[],[],[],[],[],[],[],[],[]
mMSE=[],[],[],[],[],[],[],[],[]
mDICE2=[],[],[],[],[],[],[],[],[]
i=0



for image in glob.glob('*.jpg'):    
    original = cv2.imread(image,0)
    segventanas=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/BBOX/'+image,0)
    segtoda=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ConEcuAdaptativa/BBOX/'+image,0)
    segumbral=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionRE_Articulo/BBOX/'+image,0)
    segventanas_2=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/BBOX_2/'+image,0)
    segventanas_3=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/BBOX_3/'+image,0)
    segventanas_4=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/BBOX_4/'+image,0)
    segventanas_5=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/BBOX_5/'+image,0)
    segventanas_6=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/BBOX_6/'+image,0)

    print(ssim(original, original))
    print(mse(original, original))
    print('dice2',dice(original, original))
    tabla=[original,segumbral,segtoda,segventanas,segventanas_2,segventanas_3,segventanas_4,segventanas_5,segventanas_6]
    for h in range(len(tabla)):
        mSSIM[h].append(ssim(original, tabla[h]))
        mMSE[h].append(mse(original, tabla[h]))  
        mDICE2[h].append(dice(original, tabla[h]))          
    i=i+1

import pandas as pd    
datos = {'SSIM_Original':mSSIM[0],
         'MSE_Original':mMSE[0],
         'DICE_Original':mDICE2[0],
         'SSIM_Articulo':mSSIM[1],
         'MSE_Articulo':mMSE[1],
         'DICE_Articulo':mDICE2[1],
         'SSIM_ImgComp':mSSIM[2],
         'MSE_ImgComp':mMSE[2],
         'DICE_ImgComp':mDICE2[2],
         'SSIM_ImgVen':mSSIM[3],
         'MSE_ImgVen':mMSE[3],
         'DICE_ImgVen':mDICE2[3],
         'SSIM_ImgVen2':mSSIM[4],
         'MSE_ImgVen2':mMSE[4],
         'DICE_ImgVen2':mDICE2[4],
         'SSIM_ImgVen3':mSSIM[5],
         'MSE_ImgVen3':mMSE[5],
         'DICE_ImgVen3':mDICE2[5],
         'SSIM_ImgVen4':mSSIM[6],
         'MSE_ImgVen4':mMSE[6],
         'DICE_ImgVen4':mDICE2[6],
         'SSIM_ImgVen5':mSSIM[7],
         'MSE_ImgVen5':mMSE[7],
         'DICE_ImgVen5':mDICE2[7],
         'SSIM_ImgVen6':mSSIM[8],
         'MSE_ImgVen6':mMSE[8],
         'DICE_ImgVen6':mDICE2[8]}
datos = pd.DataFrame(datos)
datos.to_excel('medidasSimilitudBBOX.xlsx') 
