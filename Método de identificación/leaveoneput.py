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


from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

med=[]
nombre=[]
matrizconfu=[]
exactitu=[]
precisione=[]
sensibili=[]
score=[]
valorauc=[]
especifici=[]
npar=[]

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
    tp = matriz[1, 1]
    fn = matriz[1, 0]
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
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/DM/leave/'+'ROC_Balanceo2'+ str(nombre))
#    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/RE/k-Fold/'+'ROC_BALANCEO2_A'+ str(nombre))
    plt.close()
    plt.show()
    return []

loo = LeaveOneOut()

loo.get_n_splits(X)
nup=0
for train_index, test_index in loo.split(X):
#   print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
               
#        sc = StandardScaler()
#        X_train_array = sc.fit_transform(X_train.values)
#        X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
#        X_test_array = sc.transform(X_test.values)
#        X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

        lr = LogisticRegression(solver='liblinear',multi_class='ovr')
        lr.fit(X_train, y_train)
        y_predictions=lr.predict(X_test)
        med.append(lr.score(X_test, y_test))
        npar.append(nup)
        nombre.append('Regresión')
        medidas(y_test,y_predictions,'Regresión')
        
        print('Regresión: ',lr.score(X_test, y_test))
        
        #rbf
        #"""
        Cc=[0.2,0.5,0.7,1,1.3]
        kernels=['rbf','linear','poly','sigmoid']
        J=0
        for C in Cc: 
            for ks in kernels: 
                clf=SVC(kernel=ks,C=C).fit(X_train,y_train)
                med.append(clf.score(X_test, y_test))
                nombre.append('SVM_'+str(ks)+'_C:_'+str(C))
                npar.append(nup)
                print('SVM ',ks,': ',clf.score(X_test,y_test))  
                y_predictionss=clf.predict(X_test)
                nooo='SVM'+ks+str(J)
                medidas(y_test,y_predictionss,nooo)
                J+=1
            lin_svc =LinearSVC(C=C).fit(X, y)
            med.append(lin_svc.score(X_test, y_test))
            nombre.append('LinearSVM_'+str(C))
            npar.append(nup)
            y_predictions=lin_svc.predict(X_test)
            medidas(y_test,y_predictions,'LinearSVM')
            print('SVMlinear:',lin_svc.score(X_test,y_test))
        #"""
        nuestim=[10,15,20,30,39,49,60,70]
        for ne in nuestim:  
                rf = RandomForestClassifier(n_estimators=ne)
                rf.fit(X_train, y_train)
                med.append(rf.score(X_test, y_test))
                nombre.append('Random forest_'+'_Estima:_'+str(ne))
                npar.append(nup)
                y_predictions=rf.predict(X_test)
                medidas(y_test,y_predictions,'Random forest'+str(ne))
                print('Random F: ',rf.score(X_test, y_test))
        
        from sklearn.naive_bayes import GaussianNB
        by = GaussianNB()
        by.fit(X_train, y_train)
        med.append(by.score(X_test, y_test))
        nombre.append('Bayes')
        npar.append(nup)
        y_predictions=by.predict(X_test)
        medidas(y_test,y_predictions,'Bayes')
        print('Bayes: ',by.score(X_test, y_test))
        
        from sklearn.neural_network import MLPClassifier
        funAc=['identity', 'logistic', 'tanh', 'relu']
        sol=['lbfgs', 'sgd', 'adam']
        for so in sol:
            for func in funAc:
                rn = MLPClassifier(activation=func,solver=so, alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)
                rn.fit(X, y) 
                med.append(rn.score(X_test, y_test))
                nombre.append('Red_'+str(so)+'_FuncionA:_'+str(func))
                npar.append(nup)
                y_predictions=rn.predict(X_test)
                medidas(y_test,y_predictions,'Red'+so+func)
                print('Red_',func,'_',so,':',rn.score(X_test, y_test))
                nup+=1 
#%%

datos = {'Numeropart':npar,
         'Tipo cla': nombre,
         'Score':med,
         'F1-score':score,
         'Matriz de Confusión':matrizconfu,
         'Exactitud':exactitu,
         'Precisión':precisione,
         'Sensibilidad':sensibili,
         'Especificidad':especifici,
         'AUC':valorauc}

datos = pd.DataFrame(datos)
datos.to_excel('leave_one_out_DM_binaria_balanceo2b.xlsx') 



