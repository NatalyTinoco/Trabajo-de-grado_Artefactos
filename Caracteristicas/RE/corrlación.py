# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:09:52 2019

@author: Nataly
"""
import pandas as pd 
import seaborn as sns
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Caracteristicas\DM\Caracateristicas_NO_H(S)V_300x300.xlsx'
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Caracteristicas\RE\todas.xlsx'
datos= pd.read_excel(file)
corr1=datos.corr()
datos = pd.DataFrame(corr1)
datos = pd.DataFrame(datos)
datos.to_excel('CorrelacióncaracteristicasRE.xlsx')    

#X=datos.drop('Clase',axis=1)


##Clase 0
#clase0=datos[0:3852].drop('Clase',axis=1)
#corr0=clase0.corr()
#ma=sns.heatmap(corr0)
#ma.set_title("Clase 0 (DM y NO)")
#
##Clase 1
#clase1=datos[3853::].drop('Clase',axis=1)
#corr1=clase1.corr()
#ma=sns.heatmap(corr1)
#ma.set_title("Clase 1 (RE)")

