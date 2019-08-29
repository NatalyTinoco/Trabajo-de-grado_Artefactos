# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:27:13 2019

@author: Nataly
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
import glob

#similitud=np.zeros((307,16))
#mSSIM=np.zeros((307,16))
#mMSE=np.zeros((307,16))
#mDICE1=np.zeros((307,16))
#mDICE2=np.zeros((307,16))
mSSIM=[],[],[],[],[],[],[],[],[]
mMSE=[],[],[],[],[],[],[],[],[]
mDICE2=[],[],[],[],[],[],[],[],[]
fileTT=[]
read=[]
i=0

workbook = xlrd.open_workbook("segentadasSI.xlsx")
sheet_names = workbook.sheet_names()
xl_sheet = workbook.sheet_by_name(sheet_names[0])

for col in range(xl_sheet.ncols):
    ttfile = xl_sheet.cell_value(1, col)
    if ttfile != '':
        ttfile=ttfile+'tif'
        fileTT.append(ttfile)

for image in glob.glob('*.tif'):    
    if image not in fileTT:
        read.append(image)
        image = '00103.tif'
        print(image)
        original = cv2.imread(image,0)
        image2 = image.split(".")
        imageFile = image2[0]+'.jpg'
        tecnica1= umbralAdapArt(imageFile)  
        tecnica2 = sthele2006(imageFile)
        tecnica3 = ventanIDEA(imageFile)
        
        dire='./Segmentadas_Prepro/15/T1/'+imageFile
        cv2.imwrite(dire,tecnica1)
        dire='./Segmentadas_Prepro/15/T2/'+imageFile
        cv2.imwrite(dire,tecnica2)
        dire='./Segmentadas_Prepro/15/T3/'+imageFile
        cv2.imwrite(dire,tecnica3)
        
#        segventanas=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/RE/'+image2[0]+'.jpg',0)
#        segtoda=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ConEcuAdaptativa/RE/'+image2[0]+'.jpg',0)
#        segumbral=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionRE_Articulo/RE/'+image2[0]+'.jpg',0)
#        segventanas_2=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/RE_2/'+image2[0]+'.jpg',0)
#        segventanas_3=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/RE_3/'+image2[0]+'.jpg',0)
#        segventanas_4=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/RE_4/'+image2[0]+'.jpg',0)
#        segventanas_5=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/RE_5/'+image2[0]+'.jpg',0)
#        segventanas_6=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subData/segmentacionSthele_CanalG_ventanas/RE_6/'+image2[0]+'.jpg',0)
        print(ssim(original, original))
        print(mse(original, original))
        print('dice2',dice(original, original))
        tabla=[original,tecnica1,tecnica2,tecnica3]
        for h in range(len(tabla)):
            mSSIM[h].append(ssim(original, tabla[h]))
            mMSE[h].append(mse(original, tabla[h]))  
            mDICE2[h].append(dice(original, tabla[h]))          
        i=i+1
#%%
        
for col in range(xl_sheet.ncols):
    ttfile = xl_sheet.cell_value(0, col)
    if ttfile != '':
        ttfile=ttfile+'tif'
        if ttfile not in read:
            print(ttfile)
#%%
import pandas as pd    
#datos = {'SSIM_Original':mSSIM[0],
#         'MSE_Original':mMSE[0],
#         'DICE_Original':mDICE2[0],
#         'SSIM_Articulo':mSSIM[1],
#         'MSE_Articulo':mMSE[1],
#         'DICE_Articulo':mDICE2[1],
#         'SSIM_ImgComp':mSSIM[2],
#         'MSE_ImgComp':mMSE[2],
#         'DICE_ImgComp':mDICE2[2],
#         'SSIM_ImgVen':mSSIM[3],
#         'MSE_ImgVen':mMSE[3],
#         'DICE_ImgVen':mDICE2[3],
#         'SSIM_ImgVen2':mSSIM[4],
#         'MSE_ImgVen2':mMSE[4],
#         'DICE_ImgVen2':mDICE2[4],
#         'SSIM_ImgVen3':mSSIM[5],
#         'MSE_ImgVen3':mMSE[5],
#         'DICE_ImgVen3':mDICE2[5],
#         'SSIM_ImgVen4':mSSIM[6],
#         'MSE_ImgVen4':mMSE[6],
#         'DICE_ImgVen4':mDICE2[6],
#         'SSIM_ImgVen5':mSSIM[7],
#         'MSE_ImgVen5':mMSE[7],
#         'DICE_ImgVen5':mDICE2[7],
#         'SSIM_ImgVen6':mSSIM[8],
#         'MSE_ImgVen6':mMSE[8],
#         'DICE_ImgVen6':mDICE2[8]}

datos = {'SSIM_Original':mSSIM[0],
         'MSE_Original':mMSE[0],
         'DICE_Original':mDICE2[0],
         'SSIM_T1':mSSIM[1],
         'MSE_T1':mMSE[1],
         'DICE_T1':mDICE2[1],
         'SSIM_T2':mSSIM[2],
         'MSE_T2':mMSE[2],
         'DICE_T2':mDICE2[2],
         'SSIM_T3':mSSIM[3],
         'MSE_T3':mMSE[3],
         'DICE_T3':mDICE2[3]}
datos = pd.DataFrame(datos)
datos.to_excel('medidasSimilitudRE-prepro15.xlsx') 
