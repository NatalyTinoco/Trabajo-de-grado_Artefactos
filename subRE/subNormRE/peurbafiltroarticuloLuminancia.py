# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:25:31 2019

@author: Nataly
"""

import cv2
import numpy as np
import pylab as plt 
from matplotlib import pyplot as plt

ima=cv2.imread('00000.jpg')
R,G,B=cv2.split(ima)
plt.imshow(ima)
plt.show()
nima=(np.min(ima)/np.max(ima))*ima
plt.imshow(nima*255)
plt.show()