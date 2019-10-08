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


import pandas as pd 
import cv2 
import numpy as np 
import random

#imagen Original (Sin RE)
#file='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de corrección/bbox/SinRE/00009.jpg'
file='00014.jpg'
img=cv2.imread(file)
imgoriginal=img.copy()
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
filebbox='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de corrección/bbox/nombres.xlsx'
datos= pd.read_excel(filebbox)

coordenadas=np.zeros((numero,4));
coordenadas[0:numero,0:2]=pos
j=0
while j <= (numero-1):
    p= random.randint(0, len(datos)-1)
    print(datos['nombres'][p])
    mask =cv2.imread('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de corrección/bbox/'+datos['nombres'][p],0)
    mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)  
    w,h=mask.shape()
    coordenadas[j,3]=w
    coordenadas[j,4]=h
    j +=1
    #img=np.asarray(ima)
#    cv2.rectangle(imagen_2,(x,y),(x+w,y+h),(0,255,0),2)


    
#%%
import cv2 
#import sys
#sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
f='00013'
fileseg='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/imagenesSeg/si/RE/Segmentadas/5/T1/'+f+'.jpg'

file='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/ImaRE/'+f+'.jpg'
img=cv2.imread(file)
imgoriginal=img.copy()

mask =cv2.imread(fileseg,0)
mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)


mask_img=img.copy()
imagen_2=img.copy()
for z in range(3):
    mask_img[:,:,z]=img[:,:,z]*mask
    
cv2.imshow('RE',mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
for z in range(3):
    imagen_2[:,:,z]=imagen_2[:,:,z]*(1-mask)
    
cv2.imshow('RE',imagen_2)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%% 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as myimg

#Imagen original
#image = cv2. imread ( 'taller.jpg')
#image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#plt.imshow(image)
#plt.title('Imagen Original')
#plt.show()

#elección de puntos para correción de perspectiva 
start = False
pt = np.array([(0,0)]);
pos= np.array([(0,0), (0,0), (0,0), (0,0)]);
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

    while(c<=1):
        cv2.imshow(title, img)
        #pos[(c-1):c,:2]=pt
        if cv2.waitKey(20) & 0xFF == 27:
            break
        
    cv2.destroyAllWindows()
#%%
x=pos[0,0]
y=pos[0,1]
w=50
h=50
#img=np.asarray(ima)
cv2.rectangle(imagen_2,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('RE',imagen_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

simu_img=imagen_2[int(y):int(y+h),int(x):int(x+w)].copy()
cv2.imshow('RE',simu_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for z in range(3):
    simu_img[:,:,z]=img[int(y):int(y+h),int(x):int(x+w),z]*mask[int(y):int(y+h),int(x):int(x+w)]

simu_f=imagen_2.copy()
simu_f[int(y):int(y+h),int(x):int(x+w)]=simu_img+imagen_2[int(y):int(y+h),int(x):int(x+w)]

cv2.imshow('RE',simu_f)
cv2.waitKey(0)
cv2.destroyAllWindows()