# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:22:29 2019

@author: Nataly
"""

import pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import xlrd

workbook = xlrd.open_workbook("metricasSimilitudHist.xlsx")

sheet = workbook.sheet_by_index(0)


i=0
SinN=np.zeros((166))
Maxmin=np.zeros((166))
rgb=np.zeros((166))
LCN=np.zeros((166))
Inte=np.zeros((166))
Estand=np.zeros((166))
Log=np.zeros((166))

"""#EUCLIDIANA
for fil in range(2,168):
    SinN[i] = sheet.cell_value(fil, 3)
    Maxmin[i]=sheet.cell_value(fil, 6)
    rgb[i]=sheet.cell_value(fil, 15)
    LCN[i]=sheet.cell_value(fil, 12)
    Inte[i]=sheet.cell_value(fil, 9)
    Estand[i]=sheet.cell_value(fil, 21)
    Log[i]=sheet.cell_value(fil, 18)
    
    i=i+1

plt.ion()
plt.boxplot([SinN, Maxmin,rgb,LCN,Inte,Estand,Log], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
plt.xticks([1,2,3,4,5,6,7], [str(1),str(2),str(3),str(4),str(5),str(6),str(7)], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
plt.ylabel(str(sheet.cell_value(1,3)))
#"""

"""#Bhattacharyya
i=0
for fil in range(2,168):
    SinN[i] = sheet.cell_value(fil, 2)
    Maxmin[i]=sheet.cell_value(fil, 5)
    rgb[i]=sheet.cell_value(fil, 14)
    LCN[i]=sheet.cell_value(fil, 11)
    Inte[i]=sheet.cell_value(fil, 8)
    Estand[i]=sheet.cell_value(fil, 20)
    Log[i]=sheet.cell_value(fil, 17)
    
    i=i+1

plt.ion()
plt.boxplot([SinN, Maxmin,rgb,LCN,Inte,Estand,Log], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
plt.xticks([1,2,3,4,5,6,7], [str(1),str(2),str(3),str(4),str(5),str(6),str(7)], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
plt.ylabel(str(sheet.cell_value(1,2)))
"""
i=0
for fil in range(2,168):
    SinN[i] = sheet.cell_value(fil, 1)
    Maxmin[i]=sheet.cell_value(fil, 4)
    rgb[i]=sheet.cell_value(fil, 13)
    LCN[i]=sheet.cell_value(fil, 10)
    Inte[i]=sheet.cell_value(fil, 7)
    Estand[i]=sheet.cell_value(fil, 19)
    Log[i]=sheet.cell_value(fil, 16)
    
    i=i+1

plt.ion()
plt.boxplot([SinN, Maxmin,rgb,LCN,Inte,Estand,Log], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
plt.xticks([1,2,3,4,5,6,7], [str(1),str(2),str(3),str(4),str(5),str(6),str(7)], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
plt.ylabel(str(sheet.cell_value(1,1)))



