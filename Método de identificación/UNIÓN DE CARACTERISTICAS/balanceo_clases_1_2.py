# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:13:31 2019

@author: Usuario
"""

import pandas as pd
import random
TA=['_1','_2','_3']

for TAB in TA:
        
    file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\DM'+TAB+'.xlsx'
    datos= pd.read_excel(file,sheet_name='Hoja1')
    datos=datos.astype(float).fillna(0.0)

    h=datos['Clase '].value_counts(0)
#    
#    datosT = [[datos['EntropiaSF'][0:h[0]]],[datos['sSIMN'][0:h[0]]],[datos['Clase '][0:h[0]]]]
    datosT = [[datos['EntropiaSF'][i] for i in range(h[0])],[datos['sSIMN'][i] for i in range(h[0])],[datos['Clase '][i] for i in range(h[0])]]
    
#    
#    
#    
#    datosT=[[],[],[]]
    datosRE1 = pd.read_excel(file, sheet_name='RE1')
    datosNO1 = pd.read_excel(file, sheet_name='OTRO1')
    datosRE2 = pd.read_excel(file, sheet_name='RE2')
    datosNO2 = pd.read_excel(file, sheet_name='OTRO2')
  
    sampleRE = []
    sampleNO=[]
    sampleRE2 = []
    sampleNO2=[]
    
    c=0
    while c <= 154:
        p= random.randint(0, len(datosRE1)-1)
        if p not in sampleRE:
            sampleRE.append(p)
            datosT[0].append(datosRE1['EntropiaSF'][p])
            datosT[1].append(datosRE1['sSIMN'][p])
            datosT[2].append(1)
            c +=1
    d=0
    while d <= 153:
        pq= random.randint(0, len(datosNO1)-1)
        if pq not in sampleNO:
            sampleNO.append(pq)
            datosT[0].append(datosNO1['EntropiaSF'][pq])
            datosT[1].append(datosNO1['sSIMN'][pq])
            datosT[2].append(1)
            d+=1
            
    c=0
    while c <= 153:
        p= random.randint(0, len(datosRE2)-1)
        if p not in sampleRE2:
            sampleRE2.append(p)
            datosT[0].append(datosRE2['EntropiaSF'][p])
            datosT[1].append(datosRE2['sSIMN'][p])
            datosT[2].append(1)
            c +=1
    d=0
    while d <= 154:
        pq= random.randint(0, len(datosNO2)-1)
        if pq not in sampleNO2:
            sampleNO2.append(pq)
            datosT[0].append(datosNO2['EntropiaSF'][pq])
            datosT[1].append(datosNO2['sSIMN'][pq])
            datosT[2].append(1)
            d+=1
        
#    datos={'clase':[1 for i in range(len(datosT[0]))],'EntropiaSF':datosT[0],'sSIMN': datosT[1]}
    datos={'clase':datosT[2],'EntropiaSF':datosT[0],'sSIMN': datosT[1]}
    consolidado = pd.DataFrame(datos).to_excel('binarioDM'+TAB+'.xlsx')
    
