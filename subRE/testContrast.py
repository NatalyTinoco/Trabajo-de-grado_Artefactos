# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:17:33 2019

@author: Usuario
"""

from readimg import read_img
import cv2
import xlrd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy

imgfile = read_img('00000.jpg')
hist = cv2.calcHist([imgfile], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
plt.plot(hist)
plt.show()

entr = entropy(hist)


#%%
imgfile = read_img('00000.jpg')
array = np.array(imgfile.size)