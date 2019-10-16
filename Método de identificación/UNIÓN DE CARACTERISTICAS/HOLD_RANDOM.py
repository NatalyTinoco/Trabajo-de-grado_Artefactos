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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\binaria_2_RE.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\balanceoRE_disi_var_asimetria.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\balanceoRE__entrop_ASB_curtos.xlsx'


#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binaria_2_DM.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\binarioDM_2.xlsx'
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_correlacion1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\BINARIOMASDE2.xlsx'


HOJA='Hoja2'
datos= pd.read_excel(file, sheet_name=HOJA)
datos=datos.astype(float).fillna(0.0)
y=datos['clase']
X=datos.drop('clase',axis=1)
print(datos['clase'].value_counts()) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


med=[]
nombre=[]
matrizconfu=[]
exactitu=[]
precisione=[]
sensibili=[]
score=[]
valorauc=[]
especifici=[]

ne=30
rf = RandomForestClassifier(n_estimators=ne)
rf.fit(X_train, y_train)

med.append(rf.score(X_test, y_test))
nombre.append('Random forest_'+'_Estima:_'+str(ne))
y_predictions=rf.predict(X_test)

matriz=confusion_matrix(y_test,y_predictions)
print('Matriz=', matriz)
exactitud=accuracy_score(y_test,y_predictions)
print('Exactitud=', exactitud)
precision=precision_score(y_test,y_predictions)
print('Precision=', precision)
sensibilidad=recall_score(y_test,y_predictions)
print('Sensibilidad=', sensibilidad )
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

print('Especificidad=',tn / (tn + fp))
print('Random F: ',rf.score(X_test, y_test))
print('AUC=', roc_auc )

#with open('model_pickle','wb') as f:
#    pickle.dump(rf,f)
#    
#with open('model_pickle','rb') as f:
#    mp = pickle.load(f)

#%%
print('=================================================================')

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\Caracteristicas_elegidas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\Caracteristicas_disi_var_asimetria.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\Caracteristicas_entro_asb_curtos.xlsx'


#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\caracDM_entropia_ssim.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_entropia_ssim.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\conca_mediaLH_altas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_mediaLH_altas.xlsx'



#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_asimetria_entropia.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_asimetria_entropia.xlsx'

file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\conca_disimilitud_homogeneidad.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_disimilitud_homogeneidad.xlsx'


#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conmascarac.xlsx'

datos1= pd.read_excel(file, sheet_name=HOJA)
datos1=datos1.astype(float).fillna(0.0)
yprueba=datos1['clase']
Xprueba=datos1.drop('clase',axis=1)
print(datos1['clase'].value_counts()) 
nombre.append('prueba')
y_predictions=rf.predict(Xprueba)

matriz=confusion_matrix(yprueba,y_predictions)
print('Matriz=', matriz)
exactitud=accuracy_score(yprueba,y_predictions)
print('Exactitud=', exactitud)
precision=precision_score(yprueba,y_predictions)
print('Precision=', precision)
sensibilidad=recall_score(yprueba,y_predictions)
print('Sensibilidad=', sensibilidad )
puntaje=f1_score(yprueba,y_predictions)
fpr, tpr, thresholds = roc_curve(yprueba,y_predictions)
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

print('Especificidad=',tn / (tn + fp))
print('Random F: ',rf.score(Xprueba, yprueba))
med.append(rf.score(Xprueba, yprueba))
print('AUC=', roc_auc )


#%%
print('=================================================================')


#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balanceo_elegidas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balanceo_disi_var_asimetria.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balanceo_entro_asb_curtos.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\caracDM_entropia_ssim.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_entropia_ssim.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_mediaLH_altas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\balance_mediaLH_altas.xlsx'



#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_asimetria_entropia.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_asimetria_entropia.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_disimilitud_homogeneidad.xlsx'
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\balance_disimilitud_homogeneidad.xlsx'


#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conmascarac.xlsx'

datos1= pd.read_excel(file, sheet_name=HOJA)
datos1=datos1.astype(float).fillna(0.0)
yprueba=datos1['clase']
Xprueba=datos1.drop('clase',axis=1)
print(datos1['clase'].value_counts()) 
nombre.append('prueba')
y_predictions=rf.predict(Xprueba)

matriz=confusion_matrix(yprueba,y_predictions)
print('Matriz=', matriz)
exactitud=accuracy_score(yprueba,y_predictions)
print('Exactitud=', exactitud)
precision=precision_score(yprueba,y_predictions)
print('Precision=', precision)
sensibilidad=recall_score(yprueba,y_predictions)
print('Sensibilidad=', sensibilidad )
puntaje=f1_score(yprueba,y_predictions)
fpr, tpr, thresholds = roc_curve(yprueba,y_predictions)
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

print('Especificidad=',tn / (tn + fp))
print('Random F: ',rf.score(Xprueba, yprueba))
med.append(rf.score(Xprueba, yprueba))
print('AUC=', roc_auc )

datos = {'Tipo cla': nombre,
         'Score':med,
         'F1-score':score,
         'Matriz de Confusión':matrizconfu,
         'Exactitud':exactitu,
         'Precisión':precisione,
         'Sensibilidad':sensibili,
         'Especificidad':especifici,
         'AUC':valorauc}

datos = pd.DataFrame(datos)
datos.to_excel('entrenamiento_test_RF.xlsx') 
