# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:17:53 2019

@author: Usuario
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
clasificadores =[LogisticRegression(solver='liblinear',multi_class='ovr'),
                 SVC(kernel='rbf',C=0.5),
                 RandomForestClassifier(n_estimators=15),
                 RandomForestClassifier(n_estimators=30),
                 RandomForestClassifier(n_estimators=60),
                 GaussianNB(),
                 MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)]
#%%
nomClas = ['Regresión logistica',
           'SVM',
           'Random forest_Estima:_(15)',
           'Random forest_Estima:_(30)',
           'Random forest_Estima:_(60)',
           'Bayes',
           'RED_ADAM_RELU']
#%%
yp=[],[],[],[],[],[],[]
yt=[],[],[],[],[],[],[]
xt1=[],[],[],[],[],[],[]
xt2=[],[],[],[],[],[],[]
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
        xt1[clas].append(X_test['EntropiaSF'].tolist())
        hl=clas+1
        xt2[clas].append(X_test['sSIMN'].tolist())
#        xt2[m].append(X_test)
        hs=hl+1 
        m+=1
#%%        
#xtt = pd.DataFrame(xt)
#ytt= pd.DataFrame(yt)
#ypp=pd.DataFrame(yp)
datos = pd.DataFrame({'EntropiaSF0':xt1[0][:],
                      'sSIMN0':xt2[0][:],
                      'EntropiaSF1':xt1[1][:],
                      'sSIMN1':xt2[1][:],
                      'EntropiaSF2':xt1[2][:],
                      'sSIMN2':xt2[2][:],
                      'EntropiaSF3':xt1[3][:],
                      'sSIMN3':xt2[3][:],
                      'EntropiaSF4':xt1[4][:],
                      'sSIMN4':xt2[4][:],
                      'EntropiaSF5':xt1[5][:],
                      'sSIMN5':xt2[5][:],
                      'EntropiaSF6':xt1[6][:],
                      'sSIMN6':xt2[6][:],
                      'yt0':yt[0][:],
                      'yt1':yt[1][:],
                      'yt2':yt[2][:],
                      'yt3':yt[3][:],
                      'yt4':yt[4][:],
                      'yt5':yt[5][:],
                      'yt6':yt[6][:],
                      'yp0':yp[0][:],
                      'yp1':yp[1][:],
                      'yp2':yp[2][:],
                      'yp3':yp[3][:],
                      'yp4':yp[4][:],
                      'yp5':yp[5][:],
                      'yp6':yp[6][:],})
datos = pd.DataFrame(datos)
datos.to_excel('Leave_DM_datos.xlsx')     
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
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\Leave_DM_datos.xlsx'     
datosm= pd.read_excel(file)  
x0=datosm[['EntropiaSF0','sSIMN0']]
x1=datosm[['EntropiaSF1','sSIMN1']]
x2=datosm[['EntropiaSF2','sSIMN2']]
x3=datosm[['EntropiaSF3','sSIMN3']]
x4=datosm[['EntropiaSF4','sSIMN4']]
x5=datosm[['EntropiaSF5','sSIMN5']]
x6=datosm[['EntropiaSF6','sSIMN6']]


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

x=[x0,x1,x2,x3,x4,x5,x6]
yt=[yt0,yt1,yt2,yt3,yt4,yt5,yt6]
yp=[yp0,yp1,yp2,yp3,yp4,yp5,yp6]
h=0
h1=2
#xt1=np.asarray(xt1).reshape(-1, 1)

for clas in range(7):
        nombre.append(nomClas[clas])
        print(nomClas[clas])
        med.append(clasificadores[clas].score(x[clas], yt[clas]))
        print('hola')        
        medidas(yt[clas],yp[clas],nomClas[clas])
        matriz=confusion_matrix(yt[clas],yp[clas])
        print(nomClas[clas])
#%%
datos = pd.DataFrame({'Tipo cla':nomClas,
                      'Score':med,
                      'F1-score':score,
                      'Matriz de Confusión':matrizconfu,
                      'Exactitud':exactitu,
                      'Precisión':precisione,
                      'Sensibilidad':sensibili,
                      'Especificidad':especifici,
                      'AUC':valorauc})
    

datos = pd.DataFrame(datos)
datos.to_excel('Leave_DM_binaria_balanceo2a.xlsx')     
#
#accuaracy1 = statistics.mean(accu[0])
#accuaracy2 = statistics.mean(accu[1])
#accuaracy3 = statistics.mean(accu[2])

   
#%%

   
