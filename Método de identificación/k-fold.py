# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:52:03 2019

@author: Nataly
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_entrenamiento.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_prueba.xlsx'
file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\datos_entrenamiento.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binariasolo2carac.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria.xlsx'

datos= pd.read_excel(file)
#tabla=['Clase','EntropiaSF','sSIMN']
datos=datos.astype(float).fillna(0.0)
#
#y=datos.Clase
#X=datos.drop('Clase',axis=1)
y=datos['Clase']
X=datos.drop('Clase',axis=1)
print(datos['Clase'].value_counts()) #contar cuantos datos hay por clase


from sklearn.model_selection import KFold
kf = KFold(n_splits=2,shuffle=True,random_state=2)

#for valores_x,valores_y in kf.split(X):
#    print(valores_x,valores_y)
kf = KFold(n_splits=4)
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    f_train_X, f_valid_X = X.iloc[train_index], X.iloc[test_index]
    f_train_y, f_valid_y = y.iloc[train_index], y.iloc[test_index]
