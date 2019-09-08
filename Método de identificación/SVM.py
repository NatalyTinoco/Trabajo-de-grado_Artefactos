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
datos= pd.read_excel(file)
#tabla=['Clase','EntropiaSF','sSIMN']
datos=datos.astype(float).fillna(0.0)

y=datos.Clase
X=datos.drop('Clase',axis=1)

print(datos['Clase'].value_counts()) #contar cuantos datos hay por clase


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,radom_state=42,stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)


from sklearn.svm import SVC
clf=SVC(kernel='rbf').fit(X_train,y_train)
print(clf.score(X_test,y_test))
#s_prediction = clf.predict(X_train)
#print (s_prediction)

