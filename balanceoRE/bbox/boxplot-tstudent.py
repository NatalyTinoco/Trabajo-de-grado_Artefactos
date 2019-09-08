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
file = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/RE/Características_OtrasRE.xlsx'

dfRE = pd.read_excel(file)
#filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/Caracateristicas_DM_(HSV)Grey_150x150.xlsx'
filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/DM/Características_OtrasDM.xlsx'
dfDM = pd.read_excel(filed)
#filen = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/NO/Caracateristicas_NO_(HSV)Grey_150x150.xlsx'
filen = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/NO/Características_OtrasNO.xlsx'
dfNO = pd.read_excel(filen)
#tabla=['Brillo','Luminancia','entropia','Media','Mediana','Asimetria','Curtosis','Cambio_Derivada','Max intensidad pos','L2','Desviación','Varianza','Valor Intensidad max','Valor pico int']
tabla=['mediahistH','medianahistH','asimetriahistH','curtosishistH','desviacionhistH','varianzahistH','valorintensidadH','valorpicoH','contrastH','energiH','homogeneiH','correlaciH','disiH','ASH','entropH','entropiaH','mediaH','medianaH','l2H','desviacionH','varianzaH','contrasteFraraH','brilloFraraH','mediahistS','medianahistS','desviacionhistS','varianzahistS','valorintensidadS','valorpicoS','contrastS','energiS','homogeneiS','correlaciS','disiS','ASS','entropS','entropiaS','mediaS','medianaS','l2S','desviacionS','varianzaS','contrasteFraraS','brilloFraraS','mediahistV','medianahistV','asimetriahistV','curtosishistV','desviacionhistV','varianzahistV','valorintensidadV','valorpicoV','contrastV','energiV','homogeneiV','correlaciV','disiV','ASV','entropV','entropiaV','mediaV','medianaV','l2V','desviacionV','varianzaV','contrasteFraraV','brilloFraraV','mediahistB','medianahistB','asimetriahistB','curtosishistB','desviacionhistB','varianzahistB','valorintensidadB','valorpicoB','contrastB','energiB','homogeneiB','correlaciB','disiB','ASB','entropB','entropiaB','mediaB','medianaB','l2B','desviacionB','varianzaB','contrasteFraraB']


statDRg=[]
pDRg=[]
statDNg=[]
pDNg=[]
statRNg=[]
pRNg=[]


statDRgmis=[]
pDRgmis=[]
statDNgmis=[]
pDNgmis=[]
statRNgmis=[]
pRNgmis=[]

for z in range (len(tabla)):
    DM=dfDM[tabla[z]]
    RE=dfRE[tabla[z]]
    NO=dfNO[tabla[z]]
    
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    #Calcule la prueba T para las medias de dos muestras independientes de puntajes
    statDR,pDR=tstudenin(DM,RE)
    statDRg.append(statDR)
    pDRg.append(pDR)
    
    statDN,pDN=tstudenin(DM,NO)
    statDNg.append(statDN)
    pDNg.append(pDN)
    
    statRN,pRN=tstudenin(RE,NO)
    statRNg.append(statRN)
    pRNg.append(pRN)  
    
    ##https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_rel.html
    #Calcula la prueba T en DOS muestras RELACIONADAS de puntajes, ay b
    statDRmis,pDRmis=tstudentrel(DM,RE)
    statDRgmis.append(statDRmis)
    pDRgmis.append(pDRmis)
    
    statDNmis,pDNmis=tstudentrel(DM,NO)
    statDNgmis.append(statDNmis)
    pDNgmis.append(pDNmis)
    
    statRNmis,pRNmis=tstudentrel(RE,NO)
    statRNgmis.append(statRNmis)
    pRNgmis.append(pRNmis)  
    
    f=plt.figure()
    box_plot_data=[DM,RE,NO]
    box=plt.boxplot(box_plot_data,patch_artist=True,labels=['DM','RE','Sin RE y/o DM'],)
#    plt.ylabel(tabla[z])
    plt.title(tabla[z])
    plt.grid(True) 
    colors = ['pink','lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoRE/bbox/'+'boxplot'+ str(tabla[z]))
    plt.close()
    plt.show()
import pandas as pd    
datos = {'Cracteristica':tabla,
         'DM vs RE (Stat)': statDRg,
         'DM vs NO (Stat)': statDNg,
         'RE vs NO (Stat)':statRNg,
         'DM vs RE (p)': pDRg,
         'DM vs NO (p)': pDNg,
         'RE vs NO (p)':pRNg,
         'DM vs RE (Stat)muestras RELACIONADAS ': statDRgmis,
         'DM vs NO (Stat)muestras RELACIONADAS ': statDNgmis,
         'RE vs NO (Stat)muestras RELACIONADAS ':statRNgmis,
         'DM vs RE (p)muestras RELACIONADAS ': pDRgmis,
         'DM vs NO (p)muestras RELACIONADAS ': pDNgmis,
         'RE vs NO (p)muestras RELACIONADAS ':pRNgmis}

datos = pd.DataFrame(datos)
#datos.to_excel('GLCMRES.xlsx') 
datos.to_excel('métricasSeparabilidadotrasre1.xlsx')         


 
            
 



