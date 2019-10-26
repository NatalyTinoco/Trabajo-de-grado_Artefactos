# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:15:19 2019

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
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_2_DM.xlsx'
file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'
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
#    tp = matriz[1, 1]
#    fn = matriz[1, 0]
    tp = matriz[1, 1]
    fn = matriz[1, 0]
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
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/RE/leave/'+'ROC'+nombre)
    plt.close()
    plt.show()
    return []
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
clasificadores =[LogisticRegression(solver='liblinear',multi_class='ovr'),
                 SVC(kernel='rbf',C=0.5),
                 RandomForestClassifier(n_estimators=15),
                 RandomForestClassifier(n_estimators=30),
                 GaussianNB(),
                 MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)]
#%%
nomClas = ['Regresión logistica',
           'SVM']

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
        lr=LogisticRegression(solver='liblinear',multi_class='ovr')
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        yp.append(y_pred)
        yt.append(np.asarray(y_test))
        xt1.append(X_test['contrastB'].tolist())
        xt2.append(X_test['desviacionB'].tolist())
        xt3.append(X_test['Brillo'].tolist())

#clasificadores =[]
nomClas = ['REGRESION']
yp=[]
yt=[]
xt1=[]
xt2=[]
xt3=[]
hs=0
m=0
#%%

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lr=LogisticRegression(solver='liblinear',multi_class='ovr')
    lr.fit(X_train, y_train)
#    lr=MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30,30))
#    lr = LogisticRegression(solver='liblinear',multi_class='ovr')
#    lr=GaussianNB()
#    lr.fit(X_train, y_train) 
    y_predictions=lr.predict(X_test)
#    rn=MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30,30),
#    rn.fit(X_train, y_train)
#    y_pred=rn.predict(X_test)
#    print(clas)
    yp.append(y_predictions)
    yt.append(np.asarray(y_test))
    xt1.append(X_test['contrastB'].tolist())
    xt2.append(X_test['desviacionB'].tolist())
    xt3.append(X_test['Brillo'].tolist())
    print('hola')
    print(len(xt1))
#      
#        xt2[m].append(X_test)
#%%        
#xtt = pd.DataFrame(xt)
#ytt= pd.DataFrame(yt)
#ypp=pd.DataFrame(yp)
datos = pd.DataFrame({'contrastB0':xt1[0][:],
                      'desviacionB0':xt2[0][:],
                      'Brillo0':xt3[0][:],
                      'contrastB1':xt1[1][:],
                      'desviacionB1':xt2[1][:],
                      'Brillo1':xt3[1][:],
                      'yt0':yt[0][:],
                      'yt1':yt[1][:],
                      'yp0':yp[0][:],
                      'yp1':yp[1][:],})
datos = pd.DataFrame(datos)
datos.to_excel('Leave_RE_datos2.xlsx')     
datos = pd.DataFrame({'contrastB0':xt1,
                      'desviacionB0':xt2,
                      'Brillo0':xt3,
                      'yt0':yt,
                      'yp0':yp,})
datos = pd.DataFrame(datos)
datos.to_excel('Leave_RE_datosRN.xlsx')     
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

file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\Leave_RE_datos2.xlsx'     
datosm= pd.read_excel(file)  

x0=datosm[['contrastB0','desviacionB0','Brillo0']]
x1=datosm[['contrastB1','desviacionB1','Brillo1']]
yt0=datosm['yt0']
yt1=datosm['yt1']
yp0=datosm['yp0']
yp1=datosm['yp1']


x=[x0,x1]
yt=[yt0,yt1]
yp=[yp0,yp1]
h=0
h1=2

for clas in range(7):
        nombre.append(nomClas[clas])
        print(nomClas[clas])
        med.append(clasificadores[clas].score(x[clas], yt[clas]))
        print(clas)        
        medidas(yt[clas],yp[clas],nomClas[clas])
#        matriz=confusion_matrix(yt6,yp6)
#%%
datos = pd.DataFrame({'Tipo cla': nombre,
                      'Score':med})
    
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\Leave_RE_datosRN.xlsx'     
datosm= pd.read_excel(file)  

x0=datosm[['contrastB0','desviacionB0','Brillo0']]
#x1=datosm[['contrastB1','desviacionB1','Brillo1']]
yt0=datosm['yt0']
#yt1=datosm['yt1']
yp0=datosm['yp0']
#yp1=datosm['yp1']


#x=[x0]
#yt=[yt0]
#yp=[yp0]
#h=0
#h1=2
nombre.append('REGRESIÓN')
med.append(lr.score(x0, yt0))
medidas(yt0,yp0,'RN')

#for clas in range(1):
#      
#        print(nomClas[clas])
#        med.append(clasificadores[clas].score(x[clas], yt[clas]))
#        print(clas)        
#        medidas(yt[clas],yp[clas],nomClas[clas])
#        matriz=confusion_matrix(yt6,yp6)
#%%
datos = pd.DataFrame({'Score':med,
                      'F1-score':score,
                      'Matriz de Confusión':matrizconfu,
                      'Exactitud':exactitu,
                      'Precisión':precisione,
                      'Sensibilidad':sensibili,
                      'Especificidad':especifici,
                      'AUC':valorauc})
    

datos = pd.DataFrame(datos)
datos.to_excel('Leave_RE_binaria_balanceo2a.xlsx')     

    

datos = pd.DataFrame(datos)
datos.to_excel('Leave_RE_binaria_balanceo2RN.xlsx')     
