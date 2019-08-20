# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:37:46 2019

@author: Usuario
"""

import glob
import shutil

for seg in glob.glob("*seg.jpg"):

    dire=dire='C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/balanceoRE/caracteForma/RE'+seg
    shutil.move(seg,dire)