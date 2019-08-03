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
R=(np.min(R)/np.max(R))*R
G=(np.min(G)/np.max(G))*G
B=(np.min(B)/np.max(B))*B
nima=ima.copy()
nima[:,:,0]=R
nima[:,:,1]=G
nima[:,:,2]=B
#nima=(np.min(ima)/np.max(ima))*ima

nima=nima.astype(np.uint8)
plt.imshow(nima*255)
plt.show()