# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:22:37 2019

@author: Nataly
"""

import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from correccion import suavizado, inpaintingB, inpaintingNS, inpaintingTA
import cv2
import glob
import pandas as pd
import numpy as np 
from skimage.measure import compare_ssim as ssim, mse, psnr
from sklearn.metrics import  cohen_kappa_score

mSSIM=[[],[],[],[]]
mMSE=[[],[],[],[]]
mPSNR=[[],[],[],[]]
mKAPPA=[[],[],[],[]]
i=0
corre=[suavizado,inpaintingB,inpaintingNS,inpaintingTA]
nu=18
for image in glob.glob('*.jpg'):
    file="./bbox/SinRE/"+image
    fileseg="./bbox/segmentaciones/"+image
    simula="./bbox/simulacion/"+image
    img=cv2.imread(file)
    mask =cv2.imread(fileseg,0)
    mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)  
    c=0
    for co in corre:
        if co==0:
            imgcorr=co(simula,mask,nu)
        else:
             imgcorr=co(simula,mask)
             
        mSSIM[c][i].append(ssim(img,imgcorr))
        mPSNR[c][i].append(psnr(img,imgcorr))
        mMSE[c][i].append(mse(img,imgcorr))  
        mKAPPA[c][i]=cohen_kappa_score(img,imgcorr)
        c+=1
    i+=1
datos = {'SSIM_suavizado':mSSIM[0][:],
         'MSE_suavizao': mMSE[0][:],
         'PSNR_suavizado':mPSNR[0][:],
         'KAPPA_suavizado':mKAPPA[0][:],
         'SSIM_inpaitingB':mSSIM[1][:],
         'MSE_inpaitingB': mMSE[1][:],
         'PSNR_inpaitingB':mPSNR[1][:],
         'KAPPA_inpaitingB':mKAPPA[1][:],
         'SSIM_inpaitingNS':mSSIM[2][:],
         'MSE_inpaitingNS': mMSE[2][:],
         'PSNR_inpaitingNS':mPSNR[2][:],
         'KAPPA_inpaitingNS':mKAPPA[2][:],
         'SSIM_inpaitingTA':mSSIM[3][:],
         'MSE_inpaitingTA': mMSE[3][:],
         'PSNR_inpaitingTA':mPSNR[3][:],
         'KAPPA_inpaitingTA':mKAPPA[3][:]}
conso=pd.DataFrame(datos)
conso.to_excel('medidasCorreccion.xlsx')
    
    
    