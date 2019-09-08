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
file = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/RE/Caracateristicas_RE_RG(B)_50x50.xlsx'

dfRE = pd.read_excel(file)
#filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/Caracateristicas_DM_(HSV)Grey_150x150.xlsx'
filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/DM/Caracateristicas_DM_RG(B)_50x50.xlsx'
dfDM = pd.read_excel(filed)
#filen = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/NO/Caracateristicas_NO_(HSV)Grey_150x150.xlsx'
filen = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/NO/Caracateristicas_NO_RG(B)_50x50.xlsx'
dfNO = pd.read_excel(filen)
#tabla=['ContrasteTF','Energia','Homogeneidad','Correlación','Disimilitud','ASM','Entropia','ContrasteDFT','EnergiaDFT','HomogeneidadDFT','CorrelaciónDFT','DisimilitudDFT','ASMDFT',	'EntropiaDFT','ContrasteSF','EnergiaSF','HomogeneidadSF','CorrelaciónSF','DisimilitudSF','ASMSF','EntropiaSF','l2waveletLL'	,'l2waveletLH','l2waveletHL','l2waveletHH','l2altas',	'l2bajas','difwaveletLH',	'difwaveletHL',	'difwaveletHH',	'mediaLL',	'mediaLH','mediaHL','mediaHH','mediaaltas','mediabajas','varLL','varLH','varHL','varHH','varbajas','varaltas','entropiaLL','entropiaLH','entropiaHL','entropiaHH','entropiabajas',	'entropiaaltas','beta','sumas','media','mediana','destan','var','l2','varlaplacianargb','varlaplacianaV','sumasHOP',	'mediaHOP','medianaHOP','destanHOP','varHOP','l2HOP','valorintensidadHFOUR','valorpicoHFOUR','sumasHFOUR',	'mediaHFOUR','medianaHFOUR','destanHFOUR','varHFOUR','asimetriaHFOUR','kurtosHFOUR','betadmHFOUR','valorintensidadHDFT','valorpicoHDFT','sumasHDFT','mediaHDFT','medianaHDFT','destanHDFT','varHDFT','asimetriaHDFT','kurtosHDFT',	'betadmHDFT','mediaEDFT','medianaEDFT','destanEDFT','varEDFT','asimetriaEDFT','kurtosEDFT','pendientereEDFT','maxderivadaEDFT','betaDWHT','DiferenciaDWHT','errorCuadraticoDWHT','sSIMDWHT',	'betaN','DiferenciaN','errorCuadraticoN','sSIMN','brillomedia','contras','brillomediana']

tabla=['ContrasteTF','Energia','Homogeneidad','Correlación','Disimilitud','ASM','Entropia','ContrasteDFT','EnergiaDFT','HomogeneidadDFT','CorrelaciónDFT','DisimilitudDFT','ASMDFT',	'EntropiaDFT','ContrasteSF','EnergiaSF','HomogeneidadSF','CorrelaciónSF','DisimilitudSF','ASMSF','EntropiaSF','l2waveletLL'	,'l2waveletLH','l2waveletHL','l2waveletHH','l2altas',	'l2bajas','difwaveletLH',	'difwaveletHL',	'difwaveletHH',	'mediaLL',	'mediaLH','mediaHL','mediaHH','mediaaltas','mediabajas','varLL','varLH','varHL','varHH','varbajas','varaltas','entropiaLL','entropiaLH','entropiaHL','entropiaHH','entropiabajas',	'entropiaaltas','beta','sumas','media','mediana','destan','var','l2','varlaplacianargb','varlaplacianaV','sumasHOP',	'mediaHOP','destanHOP','varHOP','l2HOP','valorintensidadHFOUR','valorpicoHFOUR','sumasHFOUR',	'mediaHFOUR','destanHFOUR','varHFOUR','asimetriaHFOUR','kurtosHFOUR','betadmHFOUR','valorintensidadHDFT','valorpicoHDFT','sumasHDFT','mediaHDFT','destanHDFT','varHDFT','asimetriaHDFT','kurtosHDFT',	'betadmHDFT','mediaEDFT','medianaEDFT','destanEDFT','varEDFT','asimetriaEDFT','kurtosEDFT','pendientereEDFT','maxderivadaEDFT','betaDWHT','DiferenciaDWHT','errorCuadraticoDWHT','sSIMDWHT',	'betaN','DiferenciaN','errorCuadraticoN','sSIMN','brillomedia','contras','brillomediana']


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
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/'+'boxplot_H(S)V'+ str(tabla[z]))
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
datos.to_excel('métricasSeparabilidad_RG(B)_50x50.xlsx')         


 
            
 



