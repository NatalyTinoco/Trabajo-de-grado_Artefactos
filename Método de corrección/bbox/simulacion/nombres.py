# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:04:23 2019

@author: Nataly
"""


import glob
import pandas as pd 
nombres=[]
for image in glob.glob("*.jpg"):
    nombres.append(image)
conso=pd.DataFrame(nombres)
conso.to_excel('nombresSinRE.xlsx')