# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:38:18 2019

@author: Nataly
"""

from os import remove
import glob 
for file in glob.glob("*.jpg"):
     remove('C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/evaluación/no/'+file)