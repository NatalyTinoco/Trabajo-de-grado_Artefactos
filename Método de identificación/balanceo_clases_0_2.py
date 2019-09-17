# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:13:31 2019

@author: Usuario
"""

import pandas as pd
import random

file = r'C:\Users\Usuario\Documents\Daniela\Tesis\Trabajo-de-grado_Artefactos\Método de identificación\RE\datos_entrenamiento.xlsx'

datosDM = pd.read_excel(file, sheet_name='DM')
datosNO = pd.read_excel(file, sheet_name='NAD') 
datosT = [[],[],[],[]]
datosTNO = [[],[],[],[]]
sampleDM = []
sampleNO=[]

c=0
while c <= 1925:
    p= random.randint(0, len(datosDM)-1)
    if p not in sampleDM:
        c +=1
        sampleDM.append(p)
        datosT[0].append(datosDM['contrastB'][p])
        datosT[1].append(datosDM['desviacionB'][p])
        datosT[2].append(datosDM['Brillo'][p])
        datosT[3].append('DM')

d=0
while d <= 1925:
    pq= random.randint(0, len(datosNO)-1)
    if pq not in sampleNO:
        sampleNO.append(pq)
        datosT[0].append(datosNO['contrastB'][pq])
        datosT[1].append(datosNO['desviacionB'][pq])
        datosT[2].append(datosNO['Brillo'][pq])
        datosT[3].append('NO')
        d+=1

datos={'clase':[0 for i in range(len(datosT[0]))],'contrastB':datosT[0],'desviacionB': datosT[1],'Brillo':datosT[2]}

consolidado = pd.DataFrame(datos).to_excel('RE/datos_entrenamiento_2.xlsx')

