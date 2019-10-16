# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:13:31 2019

@author: Usuario
"""

import pandas as pd
import random
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\Caracteristicas_disi_var_asimetria.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\Caracteristicas_entro_asb_curtos.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\Caracteristicas_elegidas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\RE_disi_var_asimetria.xlsx'

file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\RE_entrop_ASB_curtos.xlsx'





datos= pd.read_excel(file,sheet_name='Hoja1')
datos=datos.astype(float).fillna(0.0)

h=datos['Clase'].value_counts()
datosT=[[],[],[],[]]

datosDM = pd.read_excel(file, sheet_name='S')
datosNO = pd.read_excel(file, sheet_name='S1') 
#datosDM2 = pd.read_excel(file, sheet_name='S2')
#datosNO2 = pd.read_excel(file, sheet_name='S3') 
sampleDM = []
sampleNO=[]
sampleDM2 = []
sampleNO2=[]

a=3851
[datosT[0].append(datos['contrastB'][i]) for i in range(a)]
[datosT[1].append(datos['desviacionB'][i]) for i in range(a)]
[datosT[2].append(datos['Brillo'][i]) for i in range(a)]
[datosT[3].append(datos['Clase'][i]) for i in range(a)]

c=0
while c <= 1926:
    p= random.randint(0, len(datosDM)-1)
    if p not in sampleDM:
        c +=1
        sampleDM.append(p)
        datosT[0].append(datosDM['contrastB'][p])
        datosT[1].append(datosDM['desviacionB'][p])
        datosT[2].append(datosDM['Brillo'][p])
        datosT[3].append(0)

d=0
while d <= 1926:
    pq= random.randint(0, len(datosNO)-1)
    if pq not in sampleNO:
        sampleNO.append(pq)
        datosT[0].append(datosNO['contrastB'][pq])
        datosT[1].append(datosNO['desviacionB'][pq])
        datosT[2].append(datosNO['Brillo'][pq])
        datosT[3].append(0)
        d+=1
#c=0
#while c <=879:
#    p= random.randint(0, len(datosDM2)-1)
#    if p not in sampleDM2:
#        c +=1
#        sampleDM2.append(p)
#        datosT[0].append(datosDM2['contrastB'][p])
#        datosT[1].append(datosDM2['desviacionB'][p])
#        datosT[2].append(datosDM2['Brillo'][p])
#        datosT[3].append(0)
#
#d=0
#while d <= 879:
#    pq= random.randint(0, len(datosNO2)-1)
#    if pq not in sampleNO2:
#        sampleNO2.append(pq)
#        datosT[0].append(datosNO2['contrastB'][pq])
#        datosT[1].append(datosNO2['desviacionB'][pq])
#        datosT[2].append(datosNO2['Brillo'][pq])
#        datosT[3].append(0)
#        d+=1
#  
[datosT[0].append(datos['contrastB'][i]) for i in range(h[0],(h[0]+h[1]))]
[datosT[1].append(datos['desviacionB'][i]) for i in range(h[0],h[0]+h[1])]
[datosT[2].append(datos['Brillo'][i]) for i in range(h[0],h[0]+h[1])]
[datosT[3].append(datos['Clase'][i]) for i in range(h[0],h[0]+h[1])]
       
datos={'Clase':datosT[3],'contrastB':datosT[0],'desviacionB': datosT[1],'Brillo':datosT[2]}    

consolidado = pd.DataFrame(datos).to_excel('balanceoRE__entrop_ASB_curtos.xlsx')

