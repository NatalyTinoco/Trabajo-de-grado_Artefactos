# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:05:57 2019

@author: Nataly
"""

from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_entrenamiento.xlsx'

file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\datos_entrenamiento.xlsx'
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
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,radom_state=42,stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#%% k-fold
import numpy as np
import pandas as pd 
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
from sklearn.model_selection import KFold
kf = KFold(n_splits=2,shuffle=True,random_state=2)

#for valores_x,valores_y in kf.split(X):
#    print(valores_x,valores_y)
kf = KFold(n_splits=4)
kf.get_n_splits(X)

print(kf)
#%%    
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
regr=linear_model.LinearRegression()
for entrenamiento_indice,prueba_indice in kf.split(X):
    print('Entrenamiento: ', entrenamiento_indice, 'Prueba: ', prueba_indice)
    X_entrenamiento,X_prueba=X[entrenamiento_indice],X[prueba_indice]
    y_entrenamiento,y_prueba=y[entrenamiento_indice],y[prueba_indice]
    regr.fit(X_entrenamiento,y_entrenamiento)
    y_pred=regr.predict(X_prueba)
    print(y_pred)
    print('Error: ',mean_squared_error(y_prueba,y_pred))
    print('El valor de r^2: ', r2_score(y_prueba,y_pred))
        
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

#%%
from sklearn.svm import SVC
clf=SVC(kernel='rbf').fit(X_train,y_train)

print(clf.score(X_test,y_test))
#s_prediction = clf.predict(X_train)
#print (s_prediction)

