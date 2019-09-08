# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:36:36 2019

@author: Nataly
"""
import cv2
import xlrd
from matplotlib import pyplot as plt

#import openpyxl
#doc = openpyxl.load_workbook('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/RE/Caracateristicas_RE_HS(V)_300x300.xlsx)
#doc.get_sheet_names()
#hoja = doc.get_sheet_by_name('Sheet1')
#hoja.title
#hoja2 = doc.get_sheet_by_name('Hoja1')
#hoja2.title
#
##beta dm
#hoja.cell(row=1,column=1).value
#betaDM=[]

#workbookDM = xlrd.open_workbook("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/DM/Caracateristicas_DM_HS(V)_300x300.xlsx")
#sheetDM = workbookDM.sheet_by_index(0)
#
#workbookNO = xlrd.open_workbook("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/NO/Caracateristicas_NO_HS(V)_300x300.xlsx")
#sheetNO = workbookNO.sheet_by_index(0)
#
#workbookRE = xlrd.open_workbook("C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/RE/Caracateristicas_RE_HS(V)_300x300.xlsx")
#sheetRE = workbookRE.sheet_by_index(0)


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

file = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/RE/Caracateristicas_RE_HS(V)_150x150.xlsx'
dfRE = pd.read_excel(file)
filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/Caracateristicas_DM_HS(V)_150x150.xlsx'
dfDM = pd.read_excel(filed)
filen = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/NO/Caracateristicas_NO_HS(V)_150x150.xlsx'
dfNO = pd.read_excel(filen)
#tabla=['ContrasteTF','Energia','Homogeneidad','Correlación','Disimilitud','ASM','Entropia','ContrasteDFT','EnergiaDFT','HomogeneidadDFT','CorrelaciónDFT','DisimilitudDFT','ASMDFT',	'EntropiaDFT','ContrasteSF','EnergiaSF','HomogeneidadSF','CorrelaciónSF','DisimilitudSF','ASMSF','EntropiaSF','l2waveletLL'	,'l2waveletLH','l2waveletHL','l2waveletHH','l2altas',	'l2bajas','difwaveletLH',	'difwaveletHL',	'difwaveletHH',	'mediaLL',	'mediaLH','mediaHL','mediaHH','mediaaltas','mediabajas','varLL','varLH','varHL','varHH','varbajas','varaltas','entropiaLL','entropiaLH','entropiaHL','entropiaHH','entropiabajas',	'entropiaaltas','beta','sumas','media','mediana','destan','var','l2','varlaplacianargb','varlaplacianaV','sumasHOP',	'mediaHOP','medianaHOP','destanHOP','varHOP','l2HOP','valorintensidadHFOUR','valorpicoHFOUR','sumasHFOUR',	'mediaHFOUR','medianaHFOUR','destanHFOUR','varHFOUR','asimetriaHFOUR','kurtosHFOUR','betadmHFOUR','valorintensidadHDFT','valorpicoHDFT','sumasHDFT','mediaHDFT','medianaHDFT','destanHDFT','varHDFT','asimetriaHDFT','kurtosHDFT',	'betadmHDFT','mediaEDFT','medianaEDFT','destanEDFT','varEDFT','asimetriaEDFT','kurtosEDFT','pendientereEDFT','maxderivadaEDFT','betaDWHT','DiferenciaDWHT','errorCuadraticoDWHT','sSIMDWHT',	'betaN','DiferenciaN','errorCuadraticoN','sSIMN','brillomedia','contras','brillomediana']


file2 = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/RE/Caracateristicas_RE_H(S)V_150x150.xlsx'
dfRE2 = pd.read_excel(file2)
filed2 = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/Caracateristicas_DM_H(S)V_150x150.xlsx'
dfDM2 = pd.read_excel(filed2)
filen2 = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/NO/Caracateristicas_NO_H(S)V_150x150.xlsx'
dfNO2 = pd.read_excel(filen2)


file3 = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/RE/Caracateristicas_RE_RG(B)_150x150.xlsx'
dfRE3 = pd.read_excel(file3)
filed3 = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/DM/Caracateristicas_DM_RG(B)_150x150.xlsx'
dfDM3= pd.read_excel(filed3)
filen3 = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/150x150/NO/Caracateristicas_NO_RG(B)_150x150.xlsx'
dfNO3 = pd.read_excel(filen3)

from scipy.stats import wilcoxon

tabla=['ContrasteTF','Energia','Homogeneidad','Correlación','Disimilitud','ASM','Entropia','ContrasteDFT','EnergiaDFT','HomogeneidadDFT','CorrelaciónDFT','DisimilitudDFT','ASMDFT',	'EntropiaDFT','ContrasteSF','EnergiaSF','HomogeneidadSF','CorrelaciónSF','DisimilitudSF','ASMSF','EntropiaSF','l2waveletLL'	,'l2waveletLH','l2waveletHL','l2waveletHH','l2altas',	'l2bajas','difwaveletLH',	'difwaveletHL',	'difwaveletHH',	'mediaLL',	'mediaLH','mediaHL','mediaHH','mediaaltas','mediabajas','varLL','varLH','varHL','varHH','varbajas','varaltas','entropiaLL','entropiaLH','entropiaHL','entropiaHH','entropiabajas',	'entropiaaltas','beta','sumas','media','mediana','destan','var','l2','varlaplacianargb','varlaplacianaV','sumasHOP','mediaHOP','destanHOP','varHOP','l2HOP','valorintensidadHFOUR','valorpicoHFOUR','sumasHFOUR',	'mediaHFOUR','destanHFOUR','varHFOUR','asimetriaHFOUR','kurtosHFOUR','betadmHFOUR','valorintensidadHDFT','valorpicoHDFT','sumasHDFT','mediaHDFT','destanHDFT','varHDFT','asimetriaHDFT','kurtosHDFT',	'betadmHDFT','mediaEDFT','medianaEDFT','destanEDFT','varEDFT','asimetriaEDFT','kurtosEDFT','pendientereEDFT','maxderivadaEDFT','betaDWHT','DiferenciaDWHT','errorCuadraticoDWHT','sSIMDWHT',	'betaN','DiferenciaN','errorCuadraticoN','sSIMN','brillomedia','contras','brillomediana']

from scipy.stats import kruskal
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

statDRgmis3=[]
pDRgmis3=[]
statDNgmis3=[]
pDNgmis3=[]
statRNgmis3=[]
pRNgmis3=[]

for z in range (len(tabla)):
    DM=dfDM[tabla[z]]
    RE=dfRE[tabla[z]]
    NO=dfNO[tabla[z]]
    DM2=dfDM2[tabla[z]]
    RE2=dfRE2[tabla[z]]
    NO2=dfNO2[tabla[z]]
    DM3=dfDM3[tabla[z]]
    RE3=dfRE3[tabla[z]]
    NO3=dfNO3[tabla[z]]
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    #Calcule la prueba T para las medias de dos muestras independientes de puntajes
#    statDR,pDR=tstudenin(DM,RE)
#    statDR,pDR= wilcoxon(DM,RE)
    statDR,pDR= kruskal(DM,RE,NO)
    statDRg.append(statDR)
    pDRg.append(pDR)
    
#    statDN,pDN=tstudenin(DM,NO)
#    statDN,pDN=wilcoxon(DM,NO)
#    statDNg.append(statDN)
#    pDNg.append(pDN)
#    
##    statRN,pRN=tstudenin(RE,NO)
#    statRN,pRN=wilcoxon(RE,NO)
#    statRNg.append(statRN)
#    pRNg.append(pRN)  
    
    ##https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_rel.html
    #Calcula la prueba T en DOS muestras RELACIONADAS de puntajes, ay b
#    statDRmis,pDRmis=tstudentrel(DM,RE)
#    statDRmis,pDRmis=wilcoxon(DM2,RE2)
    statDRmis,pDRmis=kruskal(DM2,RE2,NO2)
    statDRgmis.append(statDRmis)
    pDRgmis.append(pDRmis)
    
#    statDNmis,pDNmis=tstudentrel(DM,NO)
#    statDNmis,pDNmis=wilcoxon(DM2,NO2)
#    statDNgmis.append(statDNmis)
#    pDNgmis.append(pDNmis)
#    
##    statRNmis,pRNmis=tstudentrel(RE,NO)
#    statRNmis,pRNmis=wilcoxon(RE2,NO2)
#    statRNgmis.append(statRNmis)
#    pRNgmis.append(pRNmis)  
#    
#    statDRmis3,pDRmis3=wilcoxon(DM3,RE3)
    statDRmis3,pDRmis3=kruskal(DM3,RE3,NO3)
    statDRgmis3.append(statDRmis3)
    pDRgmis3.append(pDRmis3)
    
#    statDNmis,pDNmis=tstudentrel(DM,NO)
#    statDNmis3,pDNmis3=wilcoxon(DM3,NO3)
#    statDNgmis3.append(statDNmis3)
#    pDNgmis3.append(pDNmis3)
#    
##    statRNmis,pRNmis=tstudentrel(RE,NO)
#    statRNmis3,pRNmis3=wilcoxon(RE3,NO3)
#    statRNgmis3.append(statRNmis3)
#    pRNgmis3.append(pRNmis3)  
    
#    f=plt.figure()
#    box_plot_data=[DM,RE,NO]
#    box=plt.boxplot(box_plot_data,patch_artist=True,labels=['DM','RE','Sin RE y/o DM'],)
##    plt.ylabel(tabla[z])
#    plt.title(tabla[z])
#    plt.grid(True) 
#    colors = ['pink','lightblue', 'lightgreen']
#    for patch, color in zip(box['boxes'], colors):
#        patch.set_facecolor(color)
#    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/'+'boxplot_Estiramiento_HS(V)'+ str(tabla[z]))
#    plt.close()
#    plt.show()
import pandas as pd    
datos = {'Cracteristica':tabla,
         'DM vs RE (p)HS(V)': pDRg,
#         'DM vs NO (p)': pDNg,
#         'RE vs NO (p)':pRNg,
         'DM vs RE (Stat)V': statDRg,
#         'DM vs NO (Stat)': statDNg,
#         'RE vs NO (Stat)':statRNg,
         'DM vs RE (p) H(S)V ': pDRgmis,
#         'DM vs NO (p) ': pDNgmis,
#         'RE vs NO (p) ':pRNgmis,
         'DM vs RE (Stat)S': statDRgmis,
#         'DM vs NO (Stat)': statDNgmis,
#         'RE vs NO (Stat)':statRNgmis,
         'DM vs RE (p) RG(B) ': pDRgmis,
#         'DM vs NO (p) ': pDNgmis,
#         'RE vs NO (p) ':pRNgmis,
         'DM vs RE (Stat)B': statDRgmis,
#         'DM vs NO (Stat)': statDNgmis,
#         'RE vs NO (Stat)':statRNgmis,
         }

datos = pd.DataFrame(datos)
#datos.to_excel('GLCMRES.xlsx') 
datos.to_excel('métricaskruskalfpython_150x150.xlsx')         
#        data = pandas.read_csv(url, names=names)
#        correlations = data.corr()
        
#        table=['DM','RE','NO']
#        f=plt.figure()
#        plt.ion()
#        a=plt.boxplot([DM,RE,NO],sym = 'ko', whis = 1)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
#        plt.xticks([1,2,3], ['DM','RE','Sin RE y/o DM'], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
#        plt.ylabel(tabla[z])
#        plt.grid(True)
##        plt.show()
##        f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/'+'boxplot'+ str(nombre))
##        plt.close()
#        plt.show()


 
            
 



