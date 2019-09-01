# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:27:13 2019

@author: Nataly
"""

import sys
sys.path.insert(1,'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/funciones')

from skimage.measure import compare_ssim as ssim
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import cv2
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

mSSIM=[],[],[],[],[],[],[],[],[]
mMSE=[],[],[],[],[],[],[],[],[]
mDICE2=[],[],[],[],[],[],[],[],[]
fileTT=[]
area=[]
perimetro=[]
i=0

for image in glob.glob('*.jpg'):   
    kmeansAr=[[],[],[],[],[],[]]
    print(image)
    img2 = cv2.imread(image,0)
    imrgb = cv2.imread(image)
    copyNorm = img2.copy()
    
    imgeFile = image.split(".jpg")
    original2 = cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/imagenesSeg/si/RE/'+imgeFile[0]+".tif",0)
    
#    imNorm = cv2.normalize(img2,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
    contours,hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
#        print(c)
        cnt = contours[c]
        x,y,w,h = cv2.boundingRect(cnt)
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        #cv2.drawContours(original, [approx],-1, (0,255,0), 3)
        cv2.rectangle(imrgb,(x,y),(x+w,y+h),(0,255,0),2)
        
        area.append(cv2.contourArea(approx))
        perimetro.append(cv2.arcLength(approx,True))
        
        kmeansAr[0].append(x)
        kmeansAr[1].append(y)
        kmeansAr[2].append(w)
        kmeansAr[3].append(h)
        kmeansAr[4].append(cv2.contourArea(approx))
        kmeansAr[5].append(cv2.arcLength(approx,True))
        
    pp = cv2.cvtColor(imrgb, cv2.COLOR_BGR2RGB)
    plt.imshow(pp)
    plt.show()  
#    dire='./kmeans-Seg/'+imgeFile[0]+'-contorno.jpg'
#    cv2.imwrite(dire,imrgb)
    
    X=np.array(list(zip(kmeansAr[4],kmeansAr[5])))
    kmeans = KMeans(n_clusters=2)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_
    
    colors=["m.","r.",".c","y.","b."]
    for i in range(len(X)):
        print('Coordenada: ',X[i],'Etiqueta: ',labels[i])
        plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
    plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=150,linewidths=5,zorder=10)
    plt.show()
    
    labels2 = labels.tolist()
    grupo0 = [i for i,x in enumerate(labels2) if x == 0]
    grupo1 = [i for i,x in enumerate(labels2) if x == 1]
    
    valores =0
    valores2 =0
    for p in grupo0:
        valores = kmeansAr[4][p] + valores
    valores = valores/len(grupo0)
    
    for q in grupo1:
        valores2 = kmeansAr[4][q] + valores2
    valores2 = valores2/len(grupo1)
    
    if valores2 < valores:
        for z in grupo0:
            labels2[z]=1
        for w in grupo1:
            labels2[w]=0
            
    for m in [i for i,x in enumerate(labels2) if x == 1]:
        if kmeansAr[4][m] > 2500 and kmeansAr[4][m] > 200:
            print(kmeansAr[4][m])
            copyNorm[kmeansAr[1][m]:kmeansAr[1][m]+kmeansAr[3][m], kmeansAr[0][m]:kmeansAr[0][m]+kmeansAr[2][m]] = 0
    
    pp = cv2.cvtColor(copyNorm, cv2.COLOR_BGR2RGB)
    plt.imshow(pp)
    
#    dire='./kmeans-Seg/'+ image
#    cv2.imwrite(dire,copyNorm)
            
#            if cv2.contourArea(approx) > 4000 and cv2.arcLength(approx,True) > 300:
#                 x,y,w,h = cv2.boundingRect(cnt)
#                 f,c= copyNorm[y:y+h, x:x+w].shape
#                 copyNorm[y:y+h, x:x+w] = 0
#                 pp = cv2.cvtColor(copyNorm, cv2.COLOR_BGR2RGB)
#                 plt.imshow(pp)

 
    print(ssim(original2, original2))
    print(mse(original2, original2))
    print('dice2',dice(original2, original2))
    tabla=[original2,copyNorm]
    for h in range(len(tabla)):
        mSSIM[h].append(ssim(original2, tabla[h]))
        mMSE[h].append(mse(original2, tabla[h]))  
        mDICE2[h].append(dice(original2, tabla[h]))          

#%%
import pandas as pd    


datos = {'SSIM_Original':mSSIM[0],
         'MSE_Original':mMSE[0],
         'DICE_Original':mDICE2[0],
         'SSIM_T3':mSSIM[1],
         'MSE_T3':mMSE[1],
         'DICE_T3':mDICE2[1],}
datos = pd.DataFrame(datos)
datos.to_excel('medidasSimilitudRE-kmeans2.xlsx') 
