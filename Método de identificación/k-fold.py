# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:52:03 2019

@author: Nataly
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
        
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_entrenamiento.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_balanceo.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\binaria_2_DM.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_prueba.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\datos_entrenamiento.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binariasolo2carac.xlsx'

#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria.xlsx'
file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\RE\binaria_2_RE.xlsx'

datos= pd.read_excel(file)
#tabla=['Clase','EntropiaSF','sSIMN']
datos=datos.astype(float).fillna(0.0)
#
#y=datos.Clase
#X=datos.drop('Clase',axis=1)
y=datos['Clase']
X=datos.drop('Clase',axis=1)
print(datos['Clase'].value_counts()) #contar cuantos datos hay por clase

nuparticiones=[2,4,8,10,16,20,25]


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
#    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/DM/k-Fold/'+'ROC_balanceo2A'+ str(nombre))
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Método de identificación/RE/k-Fold/'+'ROC_BALANCEO2_A'+ str(nombre))
    plt.close()
    plt.show()
    return []
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
        ##for valores_x,valores_y in kf.split(X):
        ##    print(valores_x,valores_y)
        #kf = KFold(n_splits=4)
        kf.get_n_splits(X)
        
        #print(kf)
       
    
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
#            sc = StandardScaler()
#            X_train_array = sc.fit_transform(X_train.values)
#            X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
#            X_test_array = sc.transform(X_test.values)
#            X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)
#    
#            lr = LogisticRegression(solver='liblinear',multi_class='ovr')
#            lr.fit(X_train, y_train)
#            y_predictions=lr.predict(X_test)
#            med.append(lr.score(X_test, y_test))
#            npar.append(nup)
#            nombre.append('Regresión')
#            medidas(y_test,y_predictions,'Regresión')
#            
#            print('Regresión: ',lr.score(X_test, y_test))
#            
            #rbf
            #"""
            clf=SVC(kernel='rbf',C=0.5).fit(X_train,y_train)
            med.append(clf.score(X_test, y_test))
            nombre.append('SVM_')
            npar.append(nup)
            print('SVM ',': ',clf.score(X_test,y_test))  
            y_predictionss=clf.predict(X_test)
            nooo='SVM'
            medidas(y_test,y_predictionss,nooo)

            
#            Cc=[0.2,0.5,0.7,1,1.3]
#            kernels=['rbf','linear','poly','sigmoid']
#            J=0
#            for C in Cc: 
#                for ks in kernels: 
#                    clf=SVC(kernel=ks,C=C).fit(X_train,y_train)
#                    med.append(clf.score(X_test, y_test))
#                    nombre.append('SVM_'+str(ks)+'_C:_'+str(C))
#                    npar.append(nup)
#                    print('SVM ',ks,': ',clf.score(X_test,y_test))  
#                    y_predictionss=clf.predict(X_test)
#                    nooo='SVM'+ks+str(J)
#                    medidas(y_test,y_predictionss,nooo)
#                    J+=1
#                lin_svc =LinearSVC(C=C).fit(X, y)
#                med.append(lin_svc.score(X_test, y_test))
#                nombre.append('LinearSVM_'+str(C))
#                npar.append(nup)
#                y_predictions=lin_svc.predict(X_test)
#                medidas(y_test,y_predictions,'LinearSVM')
#                print('SVMlinear:',lin_svc.score(X_test,y_test))
            #"""
#            nuestim=[10,15,20,30,39,49,60,70]
#            for ne in nuestim:  
#                    rf = RandomForestClassifier(n_estimators=ne)
#                    rf.fit(X_train, y_train)
#                    med.append(rf.score(X_test, y_test))
#                    nombre.append('Random forest_'+'_Estima:_'+str(ne))
#                    npar.append(nup)
#                    y_predictions=rf.predict(X_test)
#                    medidas(y_test,y_predictions,'Random forest'+str(ne))
#                    print('Random F: ',rf.score(X_test, y_test))
#            
#            from sklearn.naive_bayes import GaussianNB
#            by = GaussianNB()
#            by.fit(X_train, y_train)
#            med.append(by.score(X_test, y_test))
#            nombre.append('Bayes')
#            npar.append(nup)
#            y_predictions=by.predict(X_test)
#            medidas(y_test,y_predictions,'Bayes')
#            print('Bayes: ',by.score(X_test, y_test))
#            
#            from sklearn.neural_network import MLPClassifier
#            funAc=['identity', 'logistic', 'tanh', 'relu']
#            sol=['lbfgs', 'sgd', 'adam']
#            for so in sol:
#                for func in funAc:
#                    rn = MLPClassifier(activation=func,solver=so, alpha=1e-5,hidden_layer_sizes=(30,30,30,30), random_state=1)
#                    rn.fit(X, y) 
#                    med.append(rn.score(X_test, y_test))
#                    nombre.append('Red_'+str(so)+'_FuncionA:_'+str(func))
#                    npar.append(nup)
#                    y_predictions=rn.predict(X_test)
#                    medidas(y_test,y_predictions,'Red'+so+func)
#                    print('Red_',func,'_',so,':',rn.score(X_test, y_test))
#            print('kfol____numero',lac,'====================================================================================')
##%%
    grupo2= [i for i,x in enumerate(npar) if x == 2]
    grupo4= [i for i,x in enumerate(npar) if x == 4]
    grupo8= [i for i,x in enumerate(npar) if x == 8]
    grupo10 = [i for i,x in enumerate(npar) if x == 10]
    grupo16 = [i for i,x in enumerate(npar) if x == 16]
    grupo20 = [i for i,x in enumerate(npar) if x == 20]
    grupo25 = [i for i,x in enumerate(npar) if x == 25]
#%%
    nom=['Regresión','SVM_rbf_C:_0.2','SVM_linear_C:_0.2','SVM_poly_C:_0.2','SVM_sigmoid_C:_0.2','LinearSVM_0.2','SVM_rbf_C:_0.5','SVM_linear_C:_0.5','SVM_poly_C:_0.5','SVM_sigmoid_C:_0.5','LinearSVM_0.5','SVM_rbf_C:_0.7','SVM_linear_C:_0.7','SVM_poly_C:_0.7','SVM_sigmoid_C:_0.7','LinearSVM_0.7','SVM_rbf_C:_1','SVM_linear_C:_1','SVM_poly_C:_1','SVM_sigmoid_C:_1','LinearSVM_1','SVM_rbf_C:_1.3','SVM_linear_C:_1.3','SVM_poly_C:_1.3','SVM_sigmoid_C:_1.3','LinearSVM_1.3','Random forest__Estima:_10','Random forest__Estima:_15','Random forest__Estima:_20','Random forest__Estima:_30','Random forest__Estima:_39','Random forest__Estima:_49','Random forest__Estima:_60','Random forest__Estima:_70','Bayes','Red_lbfgs_FuncionA:_identity','Red_lbfgs_FuncionA:_logistic','Red_lbfgs_FuncionA:_tanh','Red_lbfgs_FuncionA:_relu','Red_sgd_FuncionA:_identity','Red_sgd_FuncionA:_logistic','Red_sgd_FuncionA:_tanh','Red_sgd_FuncionA:_relu','Red_adam_FuncionA:_identity','Red_adam_FuncionA:_logistic','Red_adam_FuncionA:_tanh','Red_adam_FuncionA:_relu']
    
    
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
    datos.to_excel('k-fold_RE_PROMEDIO_binaria_balanceo2_A'+'run'+str(lac)+'.xlsx')     
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
    datos.to_excel('k-fold_RE_binaria_balanceo2a'+'run'+str(lac)+'.xlsx') 
#%%
nombree=[]
promedio=[],[],[],[],[],[],[],[],[],[]
h=[]
import numpy as np 
import pandas as pd 
for lac in range(10):
    file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\k-fold_RE_PROMEDIO_binaria_balanceo2_A'+'run'+str(lac)+'.xlsx'     
    datos= pd.read_excel(file)   
    for indice_fila, fila in datos.iterrows():
        promedio[lac].append(fila[3::])


#%%
promediof=[]
for g in range ( len(promedio[0])):
    pro=0
    for lac in range(10):
            pro=promedio[lac][g]+pro
    promediof.append(pro/10)
    datosm = {'prome'+str(lac):promediof}

datosm = pd.DataFrame(datosm)
datosm.to_excel('k-fold_RE_binaria_A_PROMEDIOTENFOLD.xlsx')     
#
#%%






#            promedio7.append(prome7/hola)