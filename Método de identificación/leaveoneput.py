# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:55:55 2019

@author: Nataly
"""
import numpy as np
from sklearn.model_selection import LeaveOneOut

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_entrenamiento.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_balanceo.xlsx'
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


from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

loo = LeaveOneOut()

loo.get_n_splits(X)

for train_index, test_index in loo.split(X):
#   print("TRAIN:", train_index, "TEST:", test_index)
    
   X_train, X_test = X.iloc[train_index], X.iloc[test_index]
   y_train, y_test = y.iloc[train_index], y.iloc[test_index]
   ##
#   sc = StandardScaler()
#   X_train_array = sc.fit_transform(X_train.values)
#   X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
#   X_test_array = sc.transform(X_test.values)
#   X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)
#   ##
   lr = LogisticRegression(solver='liblinear',multi_class='ovr')
   lr.fit(X_train, y_train)
   y_predictions=lr.predict(X_test)
   print('Regresión: ',lr.score(X_test, y_test)) 
   #rbf
   rf = RandomForestClassifier(n_estimators=39)
   rf.fit(X_train, y_train)
   print('Random F: ',rf.score(X_test, y_test))
#   C=0.7
#   kernels=['rbf','linear','poly','sigmoid']
#   for ks in kernels: 
#       clf=SVC(kernel=ks,C=C).fit(X_train,y_train)
#       print('SVM ',ks,': ',clf.score(X_test,y_test))
    
    #        y_predictions=rf.predict(X_test)
    #        print(classification_report(y_test,y_predictions))
     

        


