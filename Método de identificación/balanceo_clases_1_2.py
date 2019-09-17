# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:13:31 2019

@author: Usuario
"""

import pandas as pd
import random

file = r'C:\Users\Usuario\Documents\Daniela\Tesis\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_entrenamiento.xlsx'

datosRE = pd.read_excel(file, sheet_name='RE')
datosNO = pd.read_excel(file, sheet_name='NAD')
datosT = [[],[],[]]
sampleRE = []
sampleNO=[]

c=0
while c <= 154:
    p= random.randint(0, len(datosRE)-1)
    if p not in sampleRE:
        sampleRE.append(p)
        datosT[0].append(datosRE['EntropiaSF'][p])
        datosT[1].append(datosRE['sSIMN'][p])
        datosT[2].append('RE')
        c +=1
d=0
while d <= 153:
    pq= random.randint(0, len(datosNO)-1)
    if pq not in sampleNO:
        sampleNO.append(pq)
        datosT[0].append(datosNO['EntropiaSF'][pq])
        datosT[1].append(datosNO['sSIMN'][pq])
        datosT[2].append('NO')
        d+=1
        
      
datos={'clase':[1 for i in range(len(datosT[0]))],'EntropiaSF':datosT[0],'sSIMN': datosT[1]}

consolidado = pd.DataFrame(datos).to_excel('DM/datos_entrenamiento_2.xlsx')

