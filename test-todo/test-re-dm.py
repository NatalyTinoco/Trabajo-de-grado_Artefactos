# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:17:02 2019

@author: Usuario
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

file=r'C:\Users\Usuario\Documents\Daniela\Tesis\Trabajo-de-grado_Artefactos\Método de identificación\binaria_2_DM'

HOJA='Hoja3'
datos= pd.read_excel(file, sheet_name=HOJA)
datos=datos.astype(float).fillna(0.0)
y=datos['clase']
X=datos.drop('clase',axis=1)
print(datos['clase'].value_counts())

pred = [[]]
train = [[]]

kf = KFold(n_splits=8,shuffle=True,random_state=2)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    ne=15
    rf = RandomForestClassifier(n_estimators=ne)
    rf.fit(X_train, y_train)
#    nombre.append('entrenamiento')
#    med.append(rf.score(X_test, y_test))
    y_predictions=rf.predict(X_test)
    pred[0].append(y_predictions)
    train[0].append(y_test)
#    medidas(y_test,y_predictions)
    print('Random F: ',rf.score(X_test, y_test))
    
pred_ = [p for p in pred[0][]]