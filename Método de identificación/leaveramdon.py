# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:58:51 2019

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
#%%
file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_2_DM.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'
datos= pd.read_excel(file)
datos=datos.astype(float).fillna(0.0)

y=datos['Clase']
X=datos.drop('Clase',axis=1)
print(datos['Clase'].value_counts())

loo = LeaveOneOut()
loo.get_n_splits(X)

regr = linear_model.LinearRegression()
lr = linear_model.LogisticRegression(solver='liblinear',multi_class='ovr')
rf = RandomForestClassifier(n_estimators=20)
bayes = GaussianNB()
#%%
med=[]
nombre=[]
matrizconfu=[]
exactitu=[]
precisione=[]
sensibili=[]
score=[]
valorauc=[]
especifici=[]
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
clasificadores =[RandomForestClassifier(n_estimators=15),
                 RandomForestClassifier(n_estimators=30)]
#%%
nomClas = ['Random forest_Estima:_(15)',
           'Random forest_Estima:_(30)']
#%%

yp=[],[]
yt=[],[]
xt1=[],[]
xt2=[],[]
xt3=[],[]
hs=0
m=0
for train_index, test_index in loo.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    for clas in range(len(clasificadores)):
        clasificadores[clas].fit(X_train, y_train)
        y_pred = clasificadores[clas].predict(X_test)
        print(clas)
        yp[clas].append(y_pred)
        yt[clas].append(np.asarray(y_test))
#        xt1[clas].append(X_test['contrastB'].tolist())
#        hl=clas+1
#        xt2[clas].append(X_test['desviacionB'].tolist())
#        xt3[clas].append(X_test['Brillo'].tolist())
        xt1[clas].append(X_test['EntropiaSF'].tolist())
        xt2[clas].append(X_test['sSIMN'].tolist())
#        xt2[m].append(X_test)

#%%        
#xtt = pd.DataFrame(xt)
#ytt= pd.DataFrame(yt)
##ypp=pd.DataFrame(yp)
#datos = pd.DataFrame({'contrastB0':xt1[0][:],
#                      'desviacionB0':xt2[0][:],
#                      'Brillo0':xt3[0][:],
#                      'contrastB1':xt1[1][:],
#                      'desviacionB1':xt2[1][:],
#                      'Brillo1':xt3[1][:],
#                      'yt0':yt[0][:],
#                      'yt1':yt[1][:],
#                      'yp0':yp[0][:],
#                      'yp1':yp[1][:],})
##ypp=pd.DataFrame(yp)
datos = pd.DataFrame({'EntropiaSF0':xt1[0][:],
                      'sSIMN0':xt2[0][:],
                      'EntropiaSF1':xt1[1][:],
                      'sSIMN1':xt2[1][:],
                      'yt0':yt[0][:],
                      'yt1':yt[1][:],
                      'yp0':yp[0][:],
                      'yp1':yp[1][:],})
datos = pd.DataFrame(datos)
datos.to_excel('Leave_DM_datos2ra.xlsx')     
##%%
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\Leave_DM_datos.xlsx'
##file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'
#datos2= pd.read_excel(file)
#datos2=datos2.astype(float).fillna(0.0)
#
#yt=datos2['yt']
#yp=datos2['yp']
#xt=datos2['xt']
#%%
import pandas as pd 
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\Leave_DM_datos2ra.xlsx'     
datosm= pd.read_excel(file)  

#x0=datosm[['contrastB0','desviacionB0','Brillo0']]
#x1=datosm[['contrastB1','desviacionB1','Brillo1']]

x0=datosm[['EntropiaSF0','sSIMN0']]
x1=datosm[['EntropiaSF1','sSIMN1']]

yt0=datosm['yt0']
yt1=datosm['yt1']


yp0=datosm['yp0']
yp1=datosm['yp1']


x=[x0,x1]
yt=[yt0,yt1]
yp=[yp0,yp1]
h=0
h1=2

for clas in range(2):
        nombre.append(nomClas[clas])
        print(nomClas[clas])
        med.append(clasificadores[clas].score(x[clas], yt[clas]))
        print(clas)        
        medidas(yt[clas],yp[clas],nomClas[clas])
#        matriz=confusion_matrix(yt6,yp6)
#%%
datos = pd.DataFrame({#'Tipo cla': nombre,
                      'Score':med,
                      'F1-score':score,
                      'Matriz de Confusión':matrizconfu,
                      'Exactitud':exactitu,
                      'Precisión':precisione,
                      'Sensibilidad':sensibili,
                      'Especificidad':especifici,
                      'AUC':valorauc})
    

datos = pd.DataFrame(datos)
datos.to_excel('Leave_DM_binaria_balanceo2a random.xlsx')     