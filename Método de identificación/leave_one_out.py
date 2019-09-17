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

file = r'C:\Users\Usuario\Documents\Daniela\Tesis\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_2_DM.xlsx'

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
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/DM/holdOUT/'+'ROC'+ str(nombre))
    plt.close()
    plt.show()
    return []

clasificadores =[RandomForestClassifier(n_estimators=60),
                 RandomForestClassifier(n_estimators=49),
                 RandomForestClassifier(n_estimators=39),
                 GaussianNB()]
nomClas = ['Random forest_Estima:_(60)',
           'Random forest_Estima:_(49)',
           'Random forest_Estima:_(39)',
           'Bayes']

for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    for clas in range(len(clasificadores)):
        clasificadores[clas].fit(X_train, y_train)
        y_pred = clasificadores[clas].predict(X_test)
        medidas(y_test,y_pred,nomClas[clas])
        med.append(clasificadores[clas].score(X_test, y_test))
        nombre.append(nomClas[clas])
        print(str(clas))
        matriz=confusion_matrix(y_test,y_pred)
        
datos = pd.DataFrame({'Tipo cla': nombre,
                      'Score':med,
                      'F1-score':score,
                      'Matriz de Confusión':matrizconfu,
                      'Exactitud':exactitu,
                      'Precisión':precisione,
                      'Sensibilidad':sensibili,
                      'Especificidad':especifici,
                      'AUC':valorauc})
    

accuaracy1 = statistics.mean(accu[0])
accuaracy2 = statistics.mean(accu[1])
accuaracy3 = statistics.mean(accu[2])

   
#%%

   
