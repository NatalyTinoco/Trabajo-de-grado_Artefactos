# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:50:34 2019

@author: Nataly
"""

import cv2
import pylab as plt 
from matplotlib import pyplot as plt
import numpy as np 
from pylab import *
import glob

for imgfile in glob.glob("*.jpg"):
    ima=cv2.imread(imgfile)
    plt.imshow(ima)
    plt.show()
    hist = cv2.calcHist([ima],[0],None,[256],[0,255])
    plt.plot(hist)
    plt.show()