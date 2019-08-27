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

file = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/RE/Caracateristicas_RE_Estiramiento_HS(V)_300x300.xlsx'
dfRE = pd.read_excel(file)
filed = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/DM/Caracateristicas_DM_Estiramiento_HS(V)_300x300.xlsx'
dfDM = pd.read_excel(filed)
filen = r'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/NO/Caracateristicas_NO_Estiramiento_HS(V)_300x300.xlsx'
dfNO = pd.read_excel(filen)
tabla=['ContrasteTF','Energia','Homogeneidad','Correlación','Disimilitud','ASM','Entropia','ContrasteDFT','EnergiaDFT','HomogeneidadDFT','CorrelaciónDFT','DisimilitudDFT','ASMDFT',	'EntropiaDFT','ContrasteSF','EnergiaSF','HomogeneidadSF','CorrelaciónSF','DisimilitudSF','ASMSF','EntropiaSF','l2waveletLL'	,'l2waveletLH','l2waveletHL','l2waveletHH','l2altas',	'l2bajas','difwaveletLH',	'difwaveletHL',	'difwaveletHH',	'mediaLL',	'mediaLH','mediaHL','mediaHH','mediaaltas','mediabajas','varLL','varLH','varHL','varHH','varbajas','varaltas','entropiaLL','entropiaLH','entropiaHL','entropiaHH','entropiabajas',	'entropiaaltas','beta','sumas','media','mediana','destan','var','l2','varlaplacianargb','varlaplacianaV','sumasHOP',	'mediaHOP','medianaHOP','destanHOP','varHOP','l2HOP','valorintensidadHFOUR','valorpicoHFOUR','sumasHFOUR',	'mediaHFOUR','medianaHFOUR','destanHFOUR','varHFOUR','asimetriaHFOUR','kurtosHFOUR','betadmHFOUR','valorintensidadHDFT','valorpicoHDFT','sumasHDFT','mediaHDFT','medianaHDFT','destanHDFT','varHDFT','asimetriaHDFT','kurtosHDFT',	'betadmHDFT','mediaEDFT','medianaEDFT','destanEDFT','varEDFT','asimetriaEDFT','kurtosEDFT','pendientereEDFT','maxderivadaEDFT','betaDWHT','DiferenciaDWHT','errorCuadraticoDWHT','sSIMDWHT',	'betaN','DiferenciaN','errorCuadraticoN','sSIMN','brillomedia','contras','brillomediana']



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
    f.savefig('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/'+'boxplot_Estiramiento_HS(V)'+ str(tabla[z]))
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
datos.to_excel('métricasSeparabilidad_Estiramiento_HS(V)_300x300.xlsx')         
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


 
            
 



