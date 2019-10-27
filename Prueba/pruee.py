# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:46:27 2019

@author: Nataly
"""
import pandas as pd
file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Prueba\evaluación/evaluacion.xlsx'

datos= pd.read_excel(file)
datos=datos.astype(float).fillna(0.0)
#
#y=datos.Clase
#X=datos.drop('Clase',axis=1)
xtest=datos['test']
xpre=datos['prueba']
#print(datos['Clase'].value_counts()) 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

matriz=confusion_matrix(xtest,xpre)
exactitud=accuracy_score(xtest,xpre)

print ('Accuracy:', accuracy_score(xtest, xpre))
print ('F1 score:', f1_score(xtest, xpre,average='weighted'))
print ('Recall:', recall_score(xtest, xpre,average='weighted'))
print ('Precision:', precision_score(xtest, xpre,average='weighted'))
print('matriz',matriz)
print('exactitud',exactitud)


