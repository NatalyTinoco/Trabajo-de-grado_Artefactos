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
mSSIM=np.zeros((307,8))
mMSE=np.zeros((307,8))
mDICE1=np.zeros((307,8))
mDICE2=np.zeros((307,8))
i=0
original = cv2.imread("./segROI/ROI_Manuales_V2/00016.jpg",0)
img=cv2.imread("./segROI/00016.jpg",0)
amSSIM=ssim(original, img)
amMSE=mse(original, img)
a=original
b=img
a=a.reshape(-1)
b=b.reshape(-1)
amDICE1= distance.dice(a, b)
amDICE2= dice(original, img)
print('SSIM=',' ',amSSIM)
print('MSE=',' ',amMSE)
print('DICE=',' ',amDICE2)


        
for imgfile in glob.glob("*.jpg"):
    #dire='./segROI/#6/Z/'+imgfile
    print(imgfile) 
    """
    original = cv2.imread("./segROI/ROI_Manuales_V2/"+imgfile,0)
    unoB = cv2.imread("./segROI/#1/B/"+imgfile,0)
    unoV = cv2.imread("./segROI/#1/V/"+imgfile,0)
    unoZ = cv2.imread("./segROI/#1/Z/"+imgfile,0)
    dosB = cv2.imread("./segROI/#2/B/"+imgfile,0)
    dosV = cv2.imread("./segROI/#2/V/"+imgfile,0)
    dosZ = cv2.imread("./segROI/#2/Z/"+imgfile,0)                     
    tresB = cv2.imread("./segROI/#3/B/"+imgfile,0)
    tresV = cv2.imread("./segROI/#3/V/"+imgfile,0)
    tresZ = cv2.imread("./segROI/#3/Z/"+imgfile,0)
    cuatroB=cv2.imread("./segROI/#4/B/"+imgfile,0)
    cuatroV=cv2.imread("./segROI/#4/V/"+imgfile,0)
    cuatroZ=cv2.imread("./segROI/#4/Z/"+imgfile,0)
    cincoB=cv2.imread("./segROI/#5/B/"+imgfile,0)
    cincoV=cv2.imread("./segROI/#5/V/"+imgfile,0)
    cincoZ=cv2.imread("./segROI/#5/Z/"+imgfile,0)
    """
    original = cv2.imread("./segROI/ROI_Manuales_V2/"+imgfile,0)
    unoB = cv2.imread("./segROI/#5/B/"+imgfile,0)
    unoV = cv2.imread("./segROI/#5/G/"+imgfile,0)
    unoZ = cv2.imread("./segROI/#5/V/"+imgfile,0)
    dosB = cv2.imread("./segROI/#5/Z/"+imgfile,0)
    #plt.imshow(dosB)
    #plt.show()
    dosV = cv2.imread("./segROI/#5/Z1/"+imgfile,0)
    dosZ = cv2.imread("./segROI/#5/Z2/"+imgfile,0)                     
    tresB = cv2.imread("./segROI/#5/Z3/"+imgfile,0)
    #tresV = cv2.imread("./segROI/#3/V/"+imgfile,0)
    #tresZ = cv2.imread("./segROI/#3/Z/"+imgfile,0)
    #cuatroB=cv2.imread("./segROI/#4/B/"+imgfile,0)
    #cuatroV=cv2.imread("./segROI/#5/B/"+imgfile,0)
    #cuatroZ=cv2.imread("./segROI/#5/G/"+imgfile,0)
    #cincoB=cv2.imread("./segROI/#5/B/"+imgfile,0)
    #cincoV=cv2.imread("./segROI/#5/V/"+imgfile,0)
    #cincoZ=cv2.imread("./segROI/#5/Z/"+imgfile,0)
    
    #tabla=[original,unoB,unoV,unoZ,dosB,dosV,dosZ,tresB,tresV,tresZ,cuatroB,cuatroV,cuatroZ,cincoB,cincoV,cincoZ ]
    tabla=[original,unoB,unoV,unoZ,dosB,dosV,dosZ,tresB]
   
    for h in range(len(tabla)):
        print(tabla[h].shape)
        mSSIM[i,h]=ssim(original, tabla[h])
        mMSE[i,h]=mse(original, tabla[h])
        a=original
        b=tabla[h]
        a=a.reshape(-1)
        b=b.reshape(-1)
        mDICE1[i,h]= distance.dice(a, b)
        mDICE2[i,h]= dice(original, tabla[h])
              
    i=i+1
    
import openpyxl

doc = openpyxl.load_workbook('ROI.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
#table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
#table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ','BA','BB','BC','BD','BE','BF','BG','BH']
table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI']

#table = ['A','B','C']

i=0
ii=0
for a in range (int(len(table)/4)):
    #print(i)
    #print(a)
    for x in range ((len(mSSIM[:,1]))):
        hoja[table[i]+ str (x+4)]=mSSIM[x,a]
        hoja[table[i+1]+ str (x+4)]=mMSE[x,a]
        hoja[table[i+2]+ str (x+4)]=mDICE1[x,a]
        hoja[table[i+3]+ str (x+4)]=mDICE2[x,a]
        
    #print(table[i])
    #print(table[i+1])
    #print(table[i+2])
    print(x,a)
    i=(a+1)*4
    #print(a)
doc.save("ROI.xlsx")