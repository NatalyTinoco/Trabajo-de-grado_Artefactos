# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 22:31:26 2019

@author: Usuario
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle

file = r'C:\Users\Usuario\Documents\Daniela\Tesis\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_2_DM.xlsx'

datos= pd.read_excel(file)
datos=datos.astype(float).fillna(0.0)

y=datos['Clase']
X=datos.drop('Clase',axis=1)
print(datos['Clase'].value_counts())

kf = KFold(n_splits=8,shuffle=True,random_state=2)
kf.get_n_splits(X)
print(kf)

rf = RandomForestClassifier(n_estimators=15)

for train_index, test_index in kf.split(X):
    f_train_X, f_valid_X = X.iloc[train_index], X.iloc[test_index]
    f_train_y, f_valid_y = y.iloc[train_index], y.iloc[test_index]
    rf.fit(f_train_X, f_train_y)
    y_predictions=rf.predict(f_valid_X)
    exactitud=metrics.accuracy_score(f_valid_y,y_predictions)
    metrics.f1_score(f_valid_y,y_predictions)

with open('model_pickle_DM','wb') as f:
    pickle.dump(rf,f)
    
with open('model_pickle_DM','rb') as f:
    mp = pickle.load(f)