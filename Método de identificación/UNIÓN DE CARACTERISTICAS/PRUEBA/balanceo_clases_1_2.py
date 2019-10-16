# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:13:31 2019

@author: Usuario
"""

import pandas as pd
import random
#TA=['_1','_2','_3']
#
#for TAB in TA:
        
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\caracDM_entropia_ssim.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_mediaLH_altas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_asimetria_entropia.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_disimilitud_homogeneidad.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conmascarac.xlsx'

datos= pd.read_excel(file,sheet_name='Hoja1')
datos=datos.astype(float).fillna(0.0)

h=datos['clase'].value_counts(0)
#    
#datosT = [[datos['Entropia'][0:h[0]]],[datos['sSIMN'][0:h[0]]],[datos['clase'][0:h[0]]]]
datosT = [[datos['Entropia'][i] for i in range(h[0])],[datos['sSIMN'][i] for i in range(h[0])],[datos['clase'][i] for i in range(h[0])]]
#datosT = [[datos['EntropiaSF'][i] for i in range(h[0])],[datos['Clase '][i] for i in range(h[0])]]
#datosT = [[datos['sSIMN'][i] for i in range(h[1])],[datos['clase '][i] for i in range(h[1])]]
#    
#    
#    
#    datosT=[[],[],[]]
datosRE1 = pd.read_excel(file, sheet_name='Hoja2')
datosNO1 = pd.read_excel(file, sheet_name='Hoja3')
datosRE2 = pd.read_excel(file, sheet_name='Hoja4')
datosNO2 = pd.read_excel(file, sheet_name='Hoja5')
#  
sampleRE = []
sampleNO=[]
sampleRE2 = []
sampleNO2=[]

c=0
while c <= 73:
    p= random.randint(0, len(datosRE1)-1)
    if p not in sampleRE:
        sampleRE.append(p)
        datosT[0].append(datosRE1['Entropia'][p])
        datosT[1].append(datosRE1['sSIMN'][p])
        datosT[2].append(1)
        c +=1
d=0
while d <= 73:
    pq= random.randint(0, len(datosNO1)-1)
    if pq not in sampleNO:
        sampleNO.append(pq)
        datosT[0].append(datosNO1['Entropia'][pq])
        datosT[1].append(datosNO1['sSIMN'][pq])
        datosT[2].append(1)
        d+=1
        
c=0
while c <= 73:
    p= random.randint(0, len(datosRE2)-1)
    if p not in sampleRE2:
        sampleRE2.append(p)
        datosT[0].append(datosRE2['Entropia'][p])
        datosT[1].append(datosRE2['sSIMN'][p])
        datosT[2].append(1)
        c +=1
d=0
while d <= 73:
    pq= random.randint(0, len(datosNO2)-1)
    if pq not in sampleNO2:
        sampleNO2.append(pq)
        datosT[0].append(datosNO2['Entropia'][pq])
        datosT[1].append(datosNO2['sSIMN'][pq])
        datosT[2].append(1)
        d+=1
#    datos={'clase':[1 for i in range(len(datosT[0]))],'Entropia':datosT[0],'sSIMN': datosT[1]}
datos={'clase':datosT[2],'Entropia':datosT[0],'sSIMN': datosT[1]}
#datos={'clase':datosT[1],'sSIMN':datosT[0]}
#consolidado = pd.DataFrame(datos).to_excel('binarioDM'+TAB+'.xlsx')
consolidado = pd.DataFrame(datos).to_excel('balance_mediaLH_altas.xlsx')

