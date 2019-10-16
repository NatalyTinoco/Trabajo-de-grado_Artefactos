# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:22:37 2019

@author: Nataly
"""

import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from correccion import suavizado, inpaintingB, inpaintingNS, inpaintingTA
import cv2
import pandas as pd
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse

from sklearn.metrics import  cohen_kappa_score

mSSIM=[[],[],[],[]]
mMSE=[[],[],[],[]]
mPSNR=[[],[],[],[]]

i=0
corre=[suavizado,inpaintingB,inpaintingNS,inpaintingTA]
coreccion=['suavizado','inpaintingB','inpaintingNS','inpaintingTA']
nu=30

files="./bbox/simulacion/nombresSinRE.xlsx"
ila= pd.read_excel(files)
for pp in range(len(ila)):
    image=ila['n'][pp]
    file="./bbox/SinRE/"+image
    fileseg="./bbox/segmentaciones/"+image
    simufile="./bbox/simulacion/"+image
    simula=cv2.imread(simufile)
    img=cv2.imread(file)
    mask =cv2.imread(fileseg,0)
    mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)  
    c=0
    for co in range(len(corre)):
        if co==0:
            imgcorr=corre[co](simula,mask,nu)
        else:
             imgcorr=corre[co](simula,mask)
       
        cv2.imwrite('./bbox/corregidas/'+coreccion[co]+image,imgcorr)     
        mSSIM[c].append(ssim(img,imgcorr,multichannel=True))
        mPSNR[c].append(psnr(img,imgcorr))
        mMSE[c].append(mse(img,imgcorr))  
        c+=1
    i+=1
datos = {'SSIM_suavizado':mSSIM[0][:],
         'MSE_suavizao': mMSE[0][:],
         'PSNR_suavizado':mPSNR[0][:],
         'SSIM_inpaitingB':mSSIM[1][:],
         'MSE_inpaitingB': mMSE[1][:],
         'PSNR_inpaitingB':mPSNR[1][:],
         'SSIM_inpaitingNS':mSSIM[2][:],
         'MSE_inpaitingNS': mMSE[2][:],
         'PSNR_inpaitingNS':mPSNR[2][:],
         'SSIM_inpaitingTA':mSSIM[3][:],
         'MSE_inpaitingTA': mMSE[3][:],
         'PSNR_inpaitingTA':mPSNR[3][:]}
conso=pd.DataFrame(datos)
conso.to_excel('medidasCorreccion_30_10.xlsx')
    
    
    