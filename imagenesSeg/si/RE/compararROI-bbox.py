# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:44:12 2019

@author: Usuario
"""
import sys

sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
#sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from sthele2006 import sthele2006

#sys.path.insert(1,'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/funciones')

import cv2
from skimage.measure import compare_ssim as ssim
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sthele2006 import sthele2006
from ventanIDEA import ventanIDEA
from umbralAdapArt import umbralAdapArt
import glob
import pandas as pd

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

mSSIM=[],[],[],[],[],[],[],[],[]
mMSE=[],[],[],[],[],[],[],[],[]
mDICE2=[],[],[],[],[],[],[],[],[]
mkernels = pd.DataFrame()
fileTT=[]
read=[]


#kernels = ['5/','10/','15/']
#tecnicas = ['T1/','T2/','T3/']
kernels = ['5/','10/','15/']
tecnicas = ['T1/','T2/','T3/']

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
        file = image
        print(file)
        origiBbox = cv2.imread(image,0)
        im2 = cv2.imread(image)
        imNorm = cv2.normalize(origiBbox,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
#        contours,hierarchy = cv2.findContours(origiBbox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        _,contours,_ = cv2.findContours(origiBbox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        origiBbox = cv2.imread(image,0)
        im2 = cv2.imread(image)
        imNorm = cv2.normalize(origiBbox,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
        contours,hierarchy = cv2.findContours(origiBbox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        jpgIma = file.split(".")[0]
        for c in range(len(contours)):
            cnt = contours[c]
            x,y,w,h = cv2.boundingRect(cnt)
            cropBbox = origiBbox[y:y+h, x:x+w]
            qqq = cv2.rectangle(origiBbox,(x,y),(x+w,y+h),(255,0,0),2)
#            rect=cv2.minAreaRect(cnt)
#            box=cv2.boxPoints(rect)
#            nueva=cv2.drawContours(im2,[box],0,(0,0,255),2)
            #pp = cv2.cvtColor(cropBbox, cv2.COLOR_BGR2RGB)
#            qqq=cv2.resize(qqq,(500,500))
#            cv2.imshow('draw',qqq)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
#            plt.imshow(cropBbox*-1,'Greys')
#            plt.show()
            print(cropBbox.shape)
            if cropBbox.shape[0]>7 and cropBbox.shape[1]>7:
                p =5
                for k in kernels:
                    i=0
                    for t in tecnicas:
                        print ('tecnica',t)
                        print('kernel',k)
                        img=cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/imagenesSeg/si/RE/Segmentadas_Prepro/'+k+t+jpgIma+'.jpg',0) 
                        cropped=img[y:y+h, x:x+w]
#                        cropped = cv2.cvtColor(cropBbox, cv2.COLOR_BGR2RGB)
                        plt.imshow(cropped*-1,'Greys')
                        plt.show()
                        mSSIM[i].append(ssim(cropBbox, cropped))
                        mMSE[i].append(mse(cropBbox, cropped))  
                        mDICE2[i].append(dice(cropBbox, cropped))
                        i+=1
                    
                    datos = {'SSIM T1':mSSIM[0],
                             'MSE T1': mMSE[0],
                             'DICE T1': mDICE2[0],
                             'SSIM T2':mSSIM[1],
                             'MSE T2': mMSE[1],
                             'DICE T2': mDICE2[1],
                             'SSIM T3':mSSIM[2],
                             'MSE T3': mMSE[2],
                             'DICE T3': mDICE2[2]
                             }
                    conso = 'consolidado'+str(p)
                    conso=pd.DataFrame(datos)
                    conso.to_excel('medidasSimilitudRE-bbox-'+str(p)+'.xlsx')
                    p+=5
    #                    pp = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    #                    plt.imshow(pp)
            #qqq = cv2.rectangle(origiBbox,(x,y),(x+w,y+h),(0,255,0),2)
            #pp = cv2.cvtColor(cropBbox, cv2.COLOR_BGR2RGB)
            #plt.imshow(pp)
            p =5
            for k in kernels:
                i=0
                for t in tecnicas:
                    img=cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/imagenesSeg/si/RE/Segmentadas_Prepro/'+k+t+jpgIma+'.jpg',0) 
                    cropped=img[y:y+h, x:x+w]
                    mSSIM[i].append(ssim(cropBbox, cropped, multichannel=True))
                    mMSE[i].append(mse(cropBbox, cropped))  
                    mDICE2[i].append(dice(cropBbox, cropped))
                    i+=1
                
                datos = {'SSIM T1':mSSIM[0],
                         'MSE T1': mMSE[0],
                         'DICE T1': mDICE2[0],
                         'SSIM T2':mSSIM[1],
                         'MSE T2': mMSE[1],
                         'DICE T2': mDICE2[1],
                         'SSIM T3':mSSIM[2],
                         'MSE T3': mMSE[2],
                         'DICE T3': mDICE2[2]
                         }
                conso = 'consolidado'+str(p)
                conso=pd.DataFrame(datos)
                conso.to_excel('medidasSimilitudRE-bbox-'+str(p)+'.xlsx')
                p+=5
#                    pp = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
#                    plt.imshow(pp)
