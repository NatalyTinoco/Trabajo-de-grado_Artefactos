# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:37:46 2019

@author: Usuario
"""
import sys
path = "C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE"
pathTxt = "C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/TODAAS"
sys.path.insert(1, path)

import cv2
import glob
import shutil
import pandas as pd
from pathlib import Path 
from yolovoc import yolo2voc


def read_boxes(file):
    import numpy as np
    lines = []
    
    for line in file:
        line = line.strip()
        box = np.hstack(line.split()).astype(np.float)
        box[0] = int(box[0])
        lines.append(box)
    return np.array(lines)

excel = pd.read_excel('ListasubRE.xlsx',header=0 ,nrows=1)
i=0

for seg in excel.loc[0]:
    imaFile = seg+"jpg"
    text = seg.split("_seg")
    txtfile = text[0]+".txt"
    im = cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE/'+imaFile)
    file = open(pathTxt+"/"+txtfile)
    bbox = read_boxes(file)
    boxes_abs = yolo2voc(bbox, im.shape)
    re=0
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 0:
            re +=1
            cropped=im[int(y1):int(y2),int(x1):int(x2),:]
            dire='C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/balanceoRE/segmentacionesRE/'+text[0]+'-'+str(re)+'.jpg'
            cv2.imwrite(dire,cropped)
            
    i+=1
    
for s in glob.glob(path + "\*seg.jpg"):
    i+=1
    f = s.split("subRE")
    print(f)
    im = cv2.imread('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE/00000_seg.jpg')
    filetxt = excel.loc[0][i]
    boxes = read_boxes(filetxt)
    i+=1
