# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:50:56 2019

@author: Nataly
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split

        
file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\binarioDM_1.xlsx'
datos= pd.read_excel(file, sheet_name='Prueba')
datos=datos.astype(float).fillna(0.0)
y=datos['clase']
X=datos.drop('clase',axis=1)
print(datos['clase'].value_counts()) 
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

lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
y_predictions=lr.predict(X_test)
med.append(lr.score(X_test, y_test))
nombre.append('Regresión')
medidas(y_test,y_predictions,'Regresión')
print('Regresión: ',lr.score(X_test, y_test))


clf=SVC(kernel='rbf',C=0.5).fit(X_train,y_train)
med.append(clf.score(X_test, y_test))
nombre.append('SVM_rbf_C:0.5')
print('SVM : ',clf.score(X_test,y_test))  
y_predictionss=clf.predict(X_test)
medidas(y_test,y_predictionss,'SVM_rbf_C:0.5')
   

nuestim=[15,30]
for ne in nuestim:  
        rf = RandomForestClassifier(n_estimators=ne)
        rf.fit(X_train, y_train)
        med.append(rf.score(X_test, y_test))
        nombre.append('Random forest_'+'_Estima:_'+str(ne))
        y_predictions=rf.predict(X_test)
        medidas(y_test,y_predictions,'Random forest'+str(ne))
        print('Random F: ',rf.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB
by = GaussianNB()
by.fit(X_train, y_train)
med.append(by.score(X_test, y_test))
nombre.append('Bayes')
y_predictions=by.predict(X_test)
medidas(y_test,y_predictions,'Bayes')
print('Bayes: ',by.score(X_test, y_test))

from sklearn.neural_network import MLPClassifier

rn = MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)
rn.fit(X, y) 
med.append(rn.score(X_test, y_test))
nombre.append('Red_relu_adam')
y_predictions=rn.predict(X_test)
medidas(y_test,y_predictions,'Red_relu_adam')
print('Red_relu_adam:',rn.score(X_test, y_test))


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
datos.to_excel('holdOut_PRUEBAdm1.xlsx') 
    
    