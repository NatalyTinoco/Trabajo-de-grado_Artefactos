# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:53:28 2019

@author: Nataly
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split

file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_2_DM.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'


datos= pd.read_excel(file)
#tabla=['Clase','EntropiaSF','sSIMN']
datos=datos.astype(float).fillna(0.0)
#
#y=datos.Clase
#X=datos.drop('Clase',axis=1)
y=datos['Clase']
X=datos.drop('Clase',axis=1)
print(datos['Clase'].value_counts()) #contar cuantos datos hay por clase
#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#%%
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train_array = sc.fit_transform(X_train.values)
#X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
#X_test_array = sc.transform(X_test.values)
#X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

#%% clasificadores

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.metrics import roc_auc_score
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
    f=plt.figure()
    plt.plot(fpr, tpr, label='curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('ROC')
    plt.legend(loc="lower right")
#    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/DM/holdOUT/'+'ROC_BALANCEO2'+ str(nombre))
#    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/ROC_BALANCEO_DM'+ str(nombre))
    plt.close()
    plt.show()
    return []

nuestim=[15,30]
for ne in nuestim:  
        rf = RandomForestClassifier(n_estimators=ne)
        rf.fit(X_train, y_train)
        med.append(rf.score(X_test, y_test))
        nombre.append('Random forest_'+'_Estima:_'+str(ne))
        y_predictions=rf.predict(X_test)
        medidas(y_test,y_predictions,'Random forest'+str(ne))
        print('Random F: ',rf.score(X_test, y_test))


