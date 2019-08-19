# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:37:46 2019

@author: Usuario
"""
import sys
path = "C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE"
sys.path.insert(1, path)

import cv2
import glob
import shutil
import pandas as pd
from pathlib import Path 
from readboxes import read_boxes

data_folder = Path(path)

excel = pd.read_excel('ListasubRE.xlsx',header=0 ,nrows=1)
for seg in excel.loc[0]:
    print(seg)
    imageFile = seg +'jpg'
    textFile = seg +'txt'
    image = cv2.imread(path + imageFile)





file_to_open = data_folder / "ListasubRE.txt"

f = open(file_to_open)
i=0
for s in glob.glob(path + "/*seg.jpg"):
    i+=1
    print(i)
    im = cv2.imread(s)
    dire=dire='C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/balanceoRE/caracteForma/RE'+seg
    shutil.move(seg,dire)