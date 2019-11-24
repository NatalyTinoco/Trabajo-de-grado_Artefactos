# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:20:42 2019

@author: Nataly
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\binaria_2_RE.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\balanceoRE_disi_var_asimetria.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\balanceoRE__entrop_ASB_curtos.xlsx'


file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binaria_2_DM.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\binarioDM_2.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_correlacion1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\BINARIOMASDE2.xlsx'


HOJA='Hoja1'
datos= pd.read_excel(file, sheet_name=HOJA)
datos=datos.astype(float).fillna(0.0)
y=datos['Clase']
X=datos.drop('Clase',axis=1)
print(datos['Clase'].value_counts()) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
ne=30
rf = RandomForestClassifier(n_estimators=ne)
rf.fit(X_train, y_train)
y_predictions=rf.predict(X_test)
matriz=confusion_matrix(y_test,y_predictions)

fpr, tpr, thresholds = roc_curve(y_test, y_predictions[:,0])
roc_auc = auc(fpr, tpr)

fpr, tpr, thresholds = roc_curve(y_test, y_predictions[:], pos_label=2)
roc_auc = auc(fpr, tpr)
    




f=plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k-*')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('ROC')
plt.legend(loc="lower right")
f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/ROC_RE')
plt.close()
plt.show()

import pandas as pd               
datos = {'testn': y_test,
         'prueba':y_predictions}
conso=pd.DataFrame(datos)
conso.to_excel('RE.xlsx')


#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
g=plt.figure()
for i in range(len(y_test)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_predictions)
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i])
#plt.plot(fpr[1], tpr[1], label='AUC = %0.2f)' % roc_auc) 
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('ROC')
g.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/ROC_RE')
plt.close()
plt.show()

#plt.show()


