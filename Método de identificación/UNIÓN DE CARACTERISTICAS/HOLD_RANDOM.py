# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:20:42 2019

@author: Nataly
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle
    
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\binarioDM_1.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'
file = r'C:\Users\Usuario\Documents\Daniela\Tesis\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'
datos= pd.read_excel(file)
datos=datos.astype(float).fillna(0.0)
y=datos['Clase']
X=datos.drop('Clase',axis=1)
print(datos['Clase'].value_counts()) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

med=[]
nombre=[]
matrizconfu=[]
exactitu=[]
precisione=[]
sensibili=[]
score=[]
valorauc=[]
especifici=[]

def medidas(y_test,y_predictions,nombre):
    matriz=confusion_matrix(y_test,y_predictions)
    exactitud=accuracy_score(y_test,y_predictions)
    precision=precision_score(y_test,y_predictions)
    sensibilidad=recall_score(y_test,y_predictions)
    puntaje=f1_score(y_test,y_predictions)
    fpr, tpr, thresholds = roc_curve(y_test, y_predictions)
    roc_auc = auc(fpr, tpr)
    matrizconfu.append(matriz)
    exactitu.append(exactitud)
    precisione.append(precision)
    sensibili.append(sensibilidad)
    score.append(puntaje)
    valorauc.append(roc_auc)
    tn = matriz[0, 0]
    tp = matriz[1, 1]
    fn = matriz[1, 0]
    fp = matriz[0, 1]
    especifici.append(tn / (tn + fp))
    return []

ne=15
rf = RandomForestClassifier(n_estimators=ne)
rf.fit(X_train, y_train)
med.append(rf.score(X_test, y_test))
nombre.append('Random forest_'+'_Estima:_'+str(ne))
y_predictions=rf.predict(X_test)
#medidas(y_test,y_predictions,'Random forest'+str(ne))
#print('Random F: ',rf.score(X_test, y_test))

with open('model_pickle','wb') as f:
    pickle.dump(rf,f)
    
with open('model_pickle','rb') as f:
    mp = pickle.load(f)

#%%
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Prueba\cutparaDM\DM\Caracateristicas_DM_PruebAA.xlsx'
file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_prueba_TODASLASCLASES.xlsx'

datos1= pd.read_excel(file, sheet_name='hola')
datos1=datos1.astype(float).fillna(0.0)
yprueba=datos1['clase']
Xprueba=datos1.drop('clase',axis=1)

y_predictions=rf.predict(Xprueba)
J=medidas(yprueba,y_predictions,'Random forest'+str(ne))
print('Random F: ',rf.score(Xprueba, yprueba))





