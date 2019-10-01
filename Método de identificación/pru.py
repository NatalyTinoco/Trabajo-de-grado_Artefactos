# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:34:41 2019

@author: Nataly
"""
import pandas as pd
from pandas import DataFrame as df
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
import statistics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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

import pandas as pd 
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\Leave_RE_datos.xlsx'     
datosm= pd.read_excel(file)  

x0=datosm[['contrastB0','desviacionB0','Brillo0']]
x1=datosm[['contrastB1','desviacionB1','Brillo1']]
x2=datosm[['contrastB2','desviacionB2','Brillo2']]
x3=datosm[['contrastB3','desviacionB3','Brillo3']]
x4=datosm[['contrastB4','desviacionB4','Brillo4']]
x5=datosm[['contrastB5','desviacionB5','Brillo5']]
x6=datosm[['contrastB6','desviacionB6','Brillo6']]


yt0=datosm['yt0']
yt1=datosm['yt1']
yt2=datosm['yt2']
yt3=datosm['yt3']
yt4=datosm['yt4']
yt5=datosm['yt5']
yt6=datosm['yt6']

yp0=datosm['yp0']
yp1=datosm['yp1']
yp2=datosm['yp2']
yp3=datosm['yp3']
yp4=datosm['yp4']
yp5=datosm['yp5']
yp6=datosm['yp6']

x=[x0,x1,x2,x4,x5,x6]
yt=[yt0,yt1,yt2,yt4,yt5,yt6]
yp=[yp0,yp1,yp2,yp4,yp5,yp6]
h=0
h1=2
#xt1=np.asarray(xt1).reshape(-1, 1)
#%%
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
#    tp = matriz[1, 1]
#    fn = matriz[1, 0]
    fp = matriz[0, 1]
    especifici.append(tn / (tn + fp))
    f=plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/DM/leave/'+'ROC'+nombre)
    plt.close()
    plt.show()
    return []
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

regr = linear_model.LinearRegression()
lr = linear_model.LogisticRegression(solver='liblinear',multi_class='ovr')
rf = RandomForestClassifier(n_estimators=20)
bayes = GaussianNB()

clasificadores=[LogisticRegression(solver='liblinear',multi_class='ovr'),
                 SVC(kernel='rbf',C=0.5),
                 RandomForestClassifier(n_estimators=15),
                  RandomForestClassifier(n_estimators=30),
                 RandomForestClassifier(n_estimators=60),
                 GaussianNB(),
                 MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)]

nomClas = ['Regresión logistica',
           'SVM',
           'Random forest_Estima:_(15)',
           'Random forest_Estima:_(60)',
           'Bayes',
           'RED_ADAM_RELU']

for clas in range(6):
        nombre.append(nomClas[clas])
        print(nomClas[clas])
        med.append(clasificadores[clas].score(x[clas], yt[clas]))
        print(clas)        
        medidas(yt[clas],yp[clas],nomClas[clas])
#        matriz=confusion_matrix(yt6,yp6)
#%%
datos = pd.DataFrame({'Tipo cla': nombre,
                      'Score':med,
                      'F1-score':score,
                      'Matriz de Confusión':matrizconfu,
                      'Exactitud':exactitu,
                      'Precisión':precisione,
                      'Sensibilidad':sensibili,
                      'Especificidad':especifici,
                      'AUC':valorauc})
    

datos = pd.DataFrame(datos)
datos.to_excel('Leave_RE_binaria_balanceo2a.xlsx')     
#
#accuaracy1 = statistics.mean(accu[0])
#accuaracy2 = statistics.mean(accu[1])
#accuaracy3 = statistics.mean(accu[2])

   