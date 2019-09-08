# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:59:07 2019

@author: Nataly
"""

import sklearn as sk
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\datos_entrenamiento.xlsx'

file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\binaria.xlsx'
datos= pd.read_excel(file)
#tabla=['Clase','EntropiaSF','sSIMN']
datos=datos.astype(float).fillna(0.0)

y=datos.Clase
X=datos.drop('Clase',axis=1)

print(datos['Clase'].value_counts()) #contar cuantos datos hay por clase


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,radom_state=42,stratify=y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train_array = sc.fit_transform(X_train.values)
#X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
#X_test_array = sc.transform(X_test.values)
#X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
LR.predict(X.iloc[460:,:])
print(round(LR.score(X,y), 4))