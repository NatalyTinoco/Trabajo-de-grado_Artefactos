# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:36:36 2019

@author: Nataly
"""
import cv2
import xlrd
from matplotlib import pyplot as plt

def tstudentrel(a,b):   
    from scipy.stats import ttest_rel
    stat, p = ttest_rel(a, b)
    return stat,p


def tstudenin(a,b):   
    from scipy.stats import ttest_ind
    stat, p = ttest_ind(a, b)
    return stat,p

from pandas import DataFrame, read_csv
import pandas as pd 

#file = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/RE/Caracateristicas_RE_(HSV)Grey_150x150.xlsx'
file = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/imagenesSeg/si/RE/caracterForma_solo_RE.xlsx'
dfRE = pd.read_excel(file)
#filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/Caracateristicas_DM_(HSV)Grey_150x150.xlsx'
filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/imagenesSeg/si/RE/caracterForma_solo_Otros.xlsx'
dfotros = pd.read_excel(filed)


statDRg=[]
pDRg=[]
statDRgmis=[]
pDRgmis=[]



tabla=['area','perimetro']

for z in range (len(tabla)):
    DM=dfotros[tabla[z]]
    RE=dfRE[tabla[z]]
    
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    #Calcule la prueba T para las medias de dos muestras independientes de puntajes
    statDR,pDR=tstudenin(RE,DM)
    statDRg.append(statDR)
    pDRg.append(pDR)
    
    ##https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_rel.html
    #Calcula la prueba T en DOS muestras RELACIONADAS de puntajes, ay b
    statDRmis,pDRmis=tstudentrel(RE,DM)
    statDRgmis.append(statDRmis)
    pDRgmis.append(pDRmis)
    

    
    f=plt.figure()
    box_plot_data=[RE,DM]
    box=plt.boxplot(box_plot_data,patch_artist=True,labels=['RE','Otros'],)
#    plt.ylabel(tabla[z])
    plt.title(tabla[z])
    plt.grid(True) 
    colors = ['pink','lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/imagenesSeg/caracterForma'+ str(tabla[z]))
    plt.close()
    plt.show()
import pandas as pd    
datos = {'Cracteristica':tabla,
         'DM vs RE (Stat)': statDRg,
         'DM vs RE (Stat)muestras RELACIONADAS ': statDRgmis,
         'DM vs RE (p)': pDRg,
         'DM vs RE (p)muestras RELACIONADAS ': pDRgmis}

datos = pd.DataFrame(datos)
#datos.to_excel('GLCMRES.xlsx') 
datos.to_excel('métricasSeparabilidad_caractforma.xlsx')         


 
            
 



