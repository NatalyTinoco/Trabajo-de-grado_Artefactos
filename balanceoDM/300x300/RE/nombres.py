# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 01:05:20 2019

@author: Nataly
"""

import cv2
import numpy as np
import glob

from matplotlib import pyplot as plt

nombres=[]
   
for image in glob.glob('*.jpg'):
    nombres.append(image)

import pandas as pd    
datos = {'nombres':nombres}
datos = pd.DataFrame(datos)
datos.to_excel('nombresbboxRE.xlsx') 

    
    