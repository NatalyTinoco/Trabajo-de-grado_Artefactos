# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:09:32 2019

@author: Nataly
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split


    
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binaria_2_DM.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_2.xlsx'
file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_correlacion1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\BINARIOMASDE2.xlsx'
datos= pd.read_excel(file)
datos=datos.astype(float).fillna(0.0)
y=datos['clase']
X=datos.drop('clase',axis=1)
print(datos['clase'].value_counts()) 
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
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
    return []

nuparticiones=[2,4,8,10,16,20,25]


for lac in range(10):
    
    npar=[]
    med=[]
    nombre=[]
    matrizconfu=[]
    exactitu=[]
    precisione=[]
    sensibili=[]
    score=[]
    valorauc=[]
    especifici=[]
    
    grupo2=[]
    grupo4=[]
    grupo8=[]
    grupo10=[]
    grupo16=[]
    grupo20=[]
    grupo25=[]
    for nup in nuparticiones:
        print('Cantidad particiones:',nup)
        kf = KFold(n_splits=nup,shuffle=True,random_state=2)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            lr = LogisticRegression(solver='liblinear',multi_class='ovr')
            lr.fit(X_train, y_train)
            y_predictions=lr.predict(X_test)
            med.append(lr.score(X_test, y_test))
            nombre.append('Regresión')
            npar.append(nup)
            medidas(y_test,y_predictions,'Regresión')
            print('Regresión: ',lr.score(X_test, y_test))
            
            
            clf=SVC(kernel='rbf',C=0.5).fit(X_train,y_train)
            med.append(clf.score(X_test, y_test))
            nombre.append('SVM_rbf_C:0.5')
            print('SVM : ',clf.score(X_test,y_test))  
            y_predictionss=clf.predict(X_test)
            medidas(y_test,y_predictionss,'SVM_rbf_C:0.5')
            npar.append(nup)   
            
            nuestim=[15,30]
            for ne in nuestim:  
                    rf = RandomForestClassifier(n_estimators=ne)
                    rf.fit(X_train, y_train)
                    med.append(rf.score(X_test, y_test))
                    nombre.append('Random forest_'+'_Estima:_'+str(ne))
                    y_predictions=rf.predict(X_test)
                    medidas(y_test,y_predictions,'Random forest'+str(ne))
                    print('Random F: ',rf.score(X_test, y_test))
                    npar.append(nup)
            
            from sklearn.naive_bayes import GaussianNB
            by = GaussianNB()
            by.fit(X_train, y_train)
            med.append(by.score(X_test, y_test))
            nombre.append('Bayes')
            y_predictions=by.predict(X_test)
            medidas(y_test,y_predictions,'Bayes')
            print('Bayes: ',by.score(X_test, y_test))
            npar.append(nup)
            
            
            from sklearn.neural_network import MLPClassifier
            
            rn = MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)
            rn.fit(X, y) 
            med.append(rn.score(X_test, y_test))
            nombre.append('Red_relu_adam')
            y_predictions=rn.predict(X_test)
            medidas(y_test,y_predictions,'Red_relu_adam')
            print('Red_relu_adam:',rn.score(X_test, y_test))
            npar.append(nup)              
    grupo2= [i for i,x in enumerate(npar) if x == 2]
    grupo4= [i for i,x in enumerate(npar) if x == 4]
    grupo8= [i for i,x in enumerate(npar) if x == 8]
    grupo10 = [i for i,x in enumerate(npar) if x == 10]
    grupo16 = [i for i,x in enumerate(npar) if x == 16]
    grupo20 = [i for i,x in enumerate(npar) if x == 20]
    grupo25 = [i for i,x in enumerate(npar) if x == 25]
#%%
    nom=['Regresión','SVM_rbf_C:0.5','Random forest__Estima:_15','Random forest__Estima:_30','Bayes','Red_relu_adam']
    ndup1=[]
    nombree=[]
    promedio1=[]
    promedio2=[]
    promedio3=[]
    promedio4=[]
    promedio5=[]
    promedio6=[]
    promedio7=[]
    
    grupoos=[grupo2,grupo4,grupo8,grupo10,grupo16,grupo20,grupo25]
    for gr in grupoos:
            for j in nom:
                hola=0
                prome1=0
                prome2=0
                prome3=0
                prome4=0
                prome5=0
                prome6=0
                prome7=0
                for p in gr:
                    if nombre[p]==j:
                        prome1=med[p]+prome1
                        prome2=score[p]+prome2
                        prome3=exactitu[p]+prome3
                        prome4=precisione[p]+prome4
                        prome5=sensibili[p]+prome5
                        prome6=especifici[p]+prome6
                        prome7=valorauc[p]+prome7  
                        hola+=1
                        print(nombre[p],j)
                print(hola,'================================================================')
                nombree.append(j)
                ndup1.append(npar[p])
                if prome1==0:
                      promedio1.append(0)
                      promedio2.append(0)
                      promedio3.append(0)
                      promedio4.append(0)
                      promedio5.append(0)
                      promedio6.append(0)
                      promedio7.append(0)
                else:
                    promedio1.append(prome1/hola)
                    promedio2.append(prome2/hola)
                    promedio3.append(prome3/hola)
                    promedio4.append(prome4/hola)
                    promedio5.append(prome5/hola)
                    promedio6.append(prome6/hola)
                    promedio7.append(prome7/hola)
    #%%
    datos = {'Numeropart':ndup1,
             'Tipo cla': nombree,
             'Score':promedio1,
             'F1-score':promedio2,
             'Exactitud':promedio3,
             'Precisión':promedio4,
             'Sensibilidad':promedio5,
             'Especificidad':promedio6,
             'AUC':promedio7}
    
    datos = pd.DataFrame(datos)
    datos.to_excel('k-fold_binarioDM_COR=1_PROMEDIO'+'run'+str(lac)+'.xlsx')     
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
    datos.to_excel('k-fold_binarioDM_COR=1_run'+str(lac)+'.xlsx') 
#%%
nombree=[]
promedio=[],[],[],[],[],[],[],[],[],[]
h=[]
#    import numpy as np 
#    import pandas as pd 

for lac in range(10):
    file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\k-fold_binarioDM_COR=1_PROMEDIO'+'run'+str(lac)+'.xlsx'     
    datos= pd.read_excel(file)   
    for indice_fila, fila in datos.iterrows():
        promedio[lac].append(fila[3::])
promediof=[]
for g in range ( len(promedio[0])):
    pro=0
    for lac in range(10):
            pro=promedio[lac][g]+pro
    promediof.append(pro/10)
    datosm = {'prome'+str(lac):promediof}

datosm = pd.DataFrame(datosm)
datosm.to_excel('k-fold_binarioDM_COR=1_PROMEDIOTENFOLD.xlsx')     
