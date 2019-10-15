# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:53:35 2019

@author: Usuario
"""
import cv2
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from Normalizacion import normalizacionMaxMin
from rOI import ROI
from caracDM import carcDM
from ventaneo import ventaneoo

#imagePath1 = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subRE/00000.jpg'
    
with open('C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/test-todo/model_pickle_DM','rb') as f:
    mpDM = pickle.load(f)

def test_all_DM(imagePath1):
    original = cv2.imread(imagePath1)
    imNorm = normalizacionMaxMin(original)
    imDR = imNorm.copy()
    roiImage = ROI(imNorm)
    for z in range(3):
        imDR[:,:,z]=imNorm[:,:,z]*roiImage
        
    aa,bb,c = imDR.shape
    a,b,ch = imDR.shape
    tamañoa1A=300
    tamañoa1B=300
    
    predictions = []
    
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
        for ca1 in range(0,b-tamañoa1B,tamañoa1B):
            croppeda1=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, imDR)
#            plt.imshow(cv2.cvtColor(croppeda1, cv2.COLOR_BGR2RGB))
#            plt.show()
            entropia,ssimn=carcDM(croppeda1)
            carac=pd.DataFrame({'entropia':entropia,'ssimn':ssimn},index =['1'])
            pred=int(mpDM.predict(carac))
            
            predictions.append(pred)
            
    return predictions

#imagePath1 = 'C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/subDM/WL_00444.jpg'
#testing=test_all_DM(imagePath1)       
#cv2.imshow('imagen0', imacropped[2])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

