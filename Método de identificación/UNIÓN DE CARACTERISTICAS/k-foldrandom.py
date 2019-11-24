# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:17:30 2019

@author: Nataly
"""

#from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt
#
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
#import numpy as np
#from pandas import DataFrame, read_csv
import pandas as pd 
#from sklearn.model_selection import train_test_split
#
#from sklearn import metrics
#
#from sklearn.preprocessing import StandardScaler


file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binaria_2_DM.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\binarioDM_2.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\DM\binarioDM_correlacion1.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\ENTRENAMIENTO\BINARIOMASDE2.xlsx'



HOJA='Hoja3'
datos= pd.read_excel(file, sheet_name=HOJA)
datos=datos.astype(float).fillna(0.0)
y=datos['clase']
X=datos.drop('clase',axis=1)
print(datos['clase'].value_counts()) #contar cuantos datos hay por clase


med=[]
nombre=[]
matrizconfu=[]
exactitu=[]
precisione=[]
sensibili=[]
score=[]
valorauc=[]
especifici=[]
def medidas(y_test,y_predictions):
    matriz=confusion_matrix(y_test,y_predictions)
    print('Matriz=', matriz)
    exactitud=accuracy_score(y_test,y_predictions)
    print('Exactitud=', exactitud)
    precision=precision_score(y_test,y_predictions)
    print('Precision=', precision)
    sensibilidad=recall_score(y_test,y_predictions)
    print('Sensibilidad=', sensibilidad )
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
    
    print('Especificidad=',tn / (tn + fp))

    print('AUC=', roc_auc )
    return []

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
kf = KFold(n_splits=8,shuffle=True,random_state=2)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    ne=15
    rf = RandomForestClassifier(n_estimators=ne)
    rf.fit(X_train, y_train)
    y_predictions=rf.predict(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_predictions)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Media ROC (AUC = %0.2f $\pm$ %0.2f)' % (0.74, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
    
#    nombre.append('entrenamiento')
#    med.append(rf.score(X_test, y_test))
#    y_predictions=rf.predict(X_test)
#    medidas(y_test,y_predictions)
#    print('Random F: ',rf.score(X_test, y_test))

#%%
print('=================================================================')
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Prueba\cutparaDM\DM\Caracateristicas_DM_PruebAA.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_prueba_TODASLASCLASES.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\caracDM_entropia_ssim.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_entropia_ssim.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\conca_mediaLH_altas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_mediaLH_altas.xlsx'



#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_asimetria_entropia.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_asimetria_entropia.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\conca_disimilitud_homogeneidad.xlsx'
##file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_disimilitud_homogeneidad.xlsx'
#
#
##file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conmascarac.xlsx'
#
#datos1= pd.read_excel(file, sheet_name=HOJA)
#datos1=datos1.astype(float).fillna(0.0)
##yprueba=datos1['clase']
#Xprueba=datos1.drop('clase',axis=1)
#print(datos1['clase'].value_counts()) 
#nombre.append('prueba')
#y_predictions=rf.predict(Xprueba)
#
#medidas(yprueba,y_predictions)
#print('Random F: ',rf.score(Xprueba, yprueba))
#med.append(rf.score(Xprueba, yprueba))


#%%
print('=================================================================')
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Prueba\cutparaDM\DM\Caracateristicas_DM_PruebAA.xlsx'
#file = r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\DM\datos_prueba_TODASLASCLASES.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\caracDM_entropia_ssim.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_entropia_ssim.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_mediaLH_altas.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\balance_mediaLH_altas.xlsx'



#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_asimetria_entropia.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\balance_asimetria_entropia.xlsx'

#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conca_disimilitud_homogeneidad.xlsx'
#file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\DM\balance_disimilitud_homogeneidad.xlsx'
#
#
##file=r'C:\Users\Nataly\Documents\Trabajo-de-grado_Artefactos\Método de identificación\UNIÓN DE CARACTERISTICAS\PRUEBA\conmascarac.xlsx'
#
#datos1= pd.read_excel(file, sheet_name=HOJA)
#datos1=datos1.astype(float).fillna(0.0)
#yprueba1=datos1['clase']
#Xprueba1=datos1.drop('clase',axis=1)
#print(datos1['clase'].value_counts()) 
#
#y_predictions=rf.predict(Xprueba1)
#nombre.append('pruebabalanceo')
#medidas(yprueba1,y_predictions)
#print('Random F: ',rf.score(Xprueba1, yprueba1))
#
#med.append(rf.score(Xprueba1, yprueba1))
#
#datos = {'Tipo cla': nombre,
#         'Score':med,
#         'F1-score':score,
#         'Matriz de Confusión':matrizconfu,
#         'Exactitud':exactitu,
#         'Precisión':precisione,
#         'Sensibilidad':sensibili,
#         'Especificidad':especifici,
#         'AUC':valorauc}
#
#datos = pd.DataFrame(datos)
#datos.to_excel('entrenamiento_test_RF.xlsx') 

