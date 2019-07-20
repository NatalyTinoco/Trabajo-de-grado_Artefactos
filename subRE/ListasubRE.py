# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 23:25:28 2019

@author: Nataly
"""


import glob
import pylab as plt 
from matplotlib import pyplot as plt

c=0
e=0
dm=0
a=0
a1=0
a2=0

box_list=[]
#filetxt = '00005.txt'
for filetxt in glob.glob("*.jpg"):
    box_list.append(filetxt[0:len(filetxt)-3])
    c=0;
    #print("Especularidad=", e)
    #print("Desenfoque M=", dm)
    dm=0
    e=0

f = open ('ListasubRE.txt','w')
f.write( " ".join(box_list))
f.close()