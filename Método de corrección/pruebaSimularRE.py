# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:32:55 2019

@author: Nataly
"""
#from tkFileDialog import *
#from Tkinter import *
#def abrir():
#   ruta=askdirectory()
#   archivo=askopenfile()
#   archivo = open("r")
#   lines = archivo.read()
#   print (lines)
#   
#ventana=Tk()
#ventana.config(bg="black")
#ventana.geometry("500x400")
#botonAbrir=Button(ventana,text="Seleccionar archivo", command=abrir)
#botonAbrir.grid(padx=150,pady=100)
#botonCompila=Button(ventana,text="Compilar")
#botonCompila.grid(padx=210,pady=10)
#ventana.mainloop()

#%%
def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def log(img,por):
    img = (np.log(img+1)/(np.log(por+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img

import pandas as pd 
import cv2 
import numpy as np 
import random

#imagen Original (Sin RE)
file="./bbox/SinRE/AN12_96.jpg"
img=cv2.imread(file)
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
        print('POS',x,y)
        pos[(c-1):c,:2]=pt
        start = True
        ventana = 'Drawing'
        grosor = cv2.getTrackbarPos('Grosor', ventana)

        cv2.circle(param, pt, grosor, (255, 0, 255), -1)
        print (pos)
if __name__ == "__main__":

    title = 'Drawing'
     
    #image = cv2. imread ( 'taller.jpg')
    cv2.namedWindow(title)
    
    cv2.createTrackbar('Grosor', title, 5, 50, on_trackbar) 
    cv2.setMouseCallback(title, on_mouse, img)

    while(c<=numero):
        cv2.imshow(title, img)
        #pos[(c-1):c,:2]=pt
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

while j <= (numero-1):
    p= random.randint(0, len(datos)-1)
    print(datos['nombres'][p])
    mask =cv2.imread('./bbox/'+datos['nombres'][p],0)
    mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)  
    mask= cv2.resize(mask,(14,14))
    h,w=mask.shape
    coordenadas[j,2]=w
    coordenadas[j,3]=h
    y=coordenadas[j,1]
    x=coordenadas[j,0]
    
    mask_img=img_1[int(y):int(y+h),int(x):int(x+w)].copy()

    cv2.destroyAllWindows()
    imagen_2= mask_img.copy()
    for z in range(3):
        mask_img[:,:,z]=img_1[int(y):int(y+h),int(x):int(x+w),z]*mask
        imagen_2[:,:,z]=imagen_2[:,:,z]*(1-mask)
    
#    mask_img=log(mask_img,500)
    for z in range(3):
#        mask_img[:,:,z]= mask_img[:,:,z]*mask
        mask_img[:,:,z]= mask*255
#    cv2.imshow('RE',mask_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    simu_f[int(y):int(y+h),int(x):int(x+w)]=mask_img+imagen_2
#    cv2.imshow('RE',simu_f)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    simu_f=simu_f
    j +=1

    #img=np.asarray(ima)
#    cv2.rectangle(imagen_2,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('RE',simu_f)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
