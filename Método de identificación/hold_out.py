# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:13:47 2019

@author: Nataly
"""
#%%
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
#%%
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
"""
#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

#%% clasificadores
med=[]
nombre=[]
lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
med.append(lr.score(X_test, y_test))
nombre.append(lr)
print('Regresión: ',lr.score(X_test, y_test))

#rbf
C=0.5
kernels=['rbf','linear','poly','sigmoid']
for ks in kernels: 
    clf=SVC(kernel=ks,C=C).fit(X_train,y_train)
    med.append(clf.score(X_test, y_test))
    nombre.append(clf)
    print('SVM ',ks,': ',clf.score(X_test,y_test))
    
lin_svc =LinearSVC(C=C).fit(X, y)
med.append(lin_svc.score(X_test, y_test))
nombre.append(lin_svc)
print('SVMlinear:',lin_svc.score(X_test,y_test))

rf = RandomForestClassifier(n_estimators=39)
rf.fit(X_train, y_train)
med.append(rf.score(X_test, y_test))
nombre.append(rf)
print('Random F: ',rf.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB
by = GaussianNB()
by.fit(X_train, y_train)
med.append(by.score(X_test, y_test))
nombre.append(by)
print('Bayes: ',by.score(X_test, y_test))

from sklearn.neural_network import MLPClassifier
funAc=['identity', 'logistic', 'tanh', 'relu']
sol=['lbfgs', 'sgd', 'adam']
for so in sol:
    for func in funAc:
        rn = MLPClassifier(activation=func,solver=so, alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)
        rn.fit(X, y) 
        med.append(rn.score(X_test, y_test))
        nombre.append(rn)
        print('Red_',func,'_',so,':',rn.score(X_test, y_test))
#%%
valormaximo=np.max(med)
pos=med.index(valormaximo)
nom=nombre[pos]
#%%
from sklearn.metrics import classification_report

y_predictions=nom.predict(X_test)
print(classification_report(y_test,y_predictions))
 



