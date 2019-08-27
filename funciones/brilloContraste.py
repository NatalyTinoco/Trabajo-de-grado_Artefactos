# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:07:00 2019

@author: Nataly
"""
import numpy as np
def contraste(img):
    im11 = img
    #arreglo = np.array(im11.size)
    #print(im11.size)
    #total = arreglo[0] * arreglo[1]
    arreglo=im11.shape
    #arreglo=list(arreglo)
    total = arreglo[0] * arreglo[1]
    i = 0
    suma = 0
    while i < arreglo[0]:
        j = 0
        while j < arreglo[1]:
            suma = suma + im11[i, j]
            j+=1        
        i+=1
    brillo = suma / total    
    i = 0
    while i < arreglo[0]:
        j = 0
        while j < arreglo[1]:
            aux = im11[i, j] - brillo
            suma = suma + aux
            j+=1
        i+=1
    cont = suma * suma
    cont = np.sqrt(suma / total)
    contraste = cont
    #print("El contraste de la imagen es: ", contraste)
    return contraste

def brillo(img):
    im10 = img
    arreglo = np.array(im10.shape)
    total = arreglo[0] * arreglo[1]
    i = 0
    suma = 0
    while i < im10.shape[0]:
        j = 0
        while j < im10.shape[1]:
            suma = suma + im10[i, j]
            j+=1        
        i+=1
    brillo = suma / total    
    brillo = int(brillo)
    #print("El brillo de la imagen es: ", brillo)

