# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:32:55 2019

@author: Nataly
"""

def log(img,por):
    img = (np.log(img+1)/(np.log(por+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img

import pandas as pd 
import cv2 
import numpy as np 
import random

#imagen Original (Sin RE)
files="./bbox/SinRE/nombresSinRE.xlsx"
ila= pd.read_excel(files)
for pp in range(len(ila)):
    image=ila['n'][pp]
    img=cv2.imread('./bbox/SinRE/'+ila['n'][pp])
    
    img=cv2.imread('./bbox/SinRE/AN12_96.jpg')
    imgoriginal=img.copy()
    img_1=img.copy()
    #saber cuantos RE se quieren en la imagen 
    numero = int(input('Introduce el número de RE: '))
    # elegir lugar de RE
    start = False
    pt = np.array([(0,0)]);
    pos=np.zeros((numero,2));
    c=0
    def on_trackbar(value):
        pass
    def on_mouse(event, x, y, flags, param):
        global c 
        global start   
        global pt          
        if event == cv2.EVENT_LBUTTONDOWN:
            c=c+1
            pt = (x, y)
            pos[(c-1):c,:2]=pt
            start = True
            ventana = 'Drawing'
            grosor = cv2.getTrackbarPos('Grosor', ventana)
            cv2.circle(param, pt, grosor, (255, 0, 255), -1)
    if __name__ == "__main__":
        title = 'Drawing'
        cv2.namedWindow(title)
        cv2.createTrackbar('Grosor', title, 1, 50, on_trackbar) 
        cv2.setMouseCallback(title, on_mouse, img)
        while(c<=numero):
            cv2.imshow(title, img)
            if cv2.waitKey(20) & 0xFF == 27:
                break  
        cv2.destroyAllWindows()
    # bbox binarios RE
    filebbox="./bbox/nombres.xlsx"
    datos= pd.read_excel(filebbox)
    coordenadas=np.zeros((numero,4));
    coordenadas[0:numero,0:2]=pos
    j=0
    simu_f=img_1.copy()
    mask_f=img_1[:,:,1].copy()
    mask_f[:,:]=0
    while j <= (numero-1):
        p= random.randint(0, len(datos)-1)
        print(datos['nombres'][p])
        mask =cv2.imread('./bbox/'+datos['nombres'][p],0)
        mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)  
        ho,wo,co=img.shape
        print(ho,wo,co)
        jj=0
        hh,ww=mask.shape
#        tamaños=[8,10,14,13,15,17,22,25,40,50,45,42,73]
        tamaños=[hh,ww,50,36]
        while jj <= (1):
            pp= random.randint(0, len(tamaños)-1)
            pp2= random.randint(0, len(tamaños)-1)
            mask= cv2.resize(mask,(tamaños[pp],tamaños[pp2]))
            jj+=1          
        h,w=mask.shape
        coordenadas[j,2]=w
        coordenadas[j,3]=h
        y=coordenadas[j,1]
        x=coordenadas[j,0]
        mask_f[int(y):int(y+h),int(x):int(x+w)]=mask
        mask_img=img_1[int(y):int(y+h),int(x):int(x+w)].copy()
        cv2.destroyAllWindows()
        imagen_2= mask_img.copy()
        for z in range(3):
            mask_img[:,:,z]=img_1[int(y):int(y+h),int(x):int(x+w),z]*mask
            imagen_2[:,:,z]=imagen_2[:,:,z]*(1-mask)
        
        mask_img=log(mask_img,700)
        for z in range(3):
    #        mask_img[:,:,z]= mask_img[:,:,z]*mask
            mask_img[:,:,z]= mask*227
        simu_f[int(y):int(y+h),int(x):int(x+w)]=mask_img+imagen_2
        simu_f=simu_f
        j +=1
    cv2.imshow('RE',simu_f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./bbox/simulacion_2/'+image,simu_f)
    mask_f= cv2.normalize(mask_f, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)  
    cv2.imwrite('./bbox/segmentaciones_2/'+image,mask_f)
    
       
