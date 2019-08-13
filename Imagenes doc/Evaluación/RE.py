# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:59:15 2019

@author: Nataly
"""

from matplotlib import pyplot as plt
import cv2
import numpy as np
i=0
file='00000.jpg'
seg='00000_seg.jpg'
img = cv2.imread(file)
def tloga(img):
    img = (np.log(img+1)/(np.log(1+np.max(img))))*255
    img = np.array(img,dtype=np.uint8)
    return img
img=tloga(img)
img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
segima=img.copy()
imaROI=cv2.imread(seg,0)
imaROI1=imaROI.copy()
imaROI1=imaROI*-1
imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
imaROI1=cv2.normalize(imaROI1, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)



for z in range(3):
    img[:,:,z]= img[:,:,z]*(imaROI)

plt.imshow(img)
plt.show()
#plt.hist(img.ravel(),[256])
#plt.show()
for i in range(3):
    hist = cv2.calcHist([img], [i], None, [256], [1, 256])

plt.plot(hist)
plt.show()

   
for z in range(3):
    segima[:,:,z]= segima[:,:,z]*imaROI1

plt.imshow(segima)
plt.show()

for i in range(3):
    hist = cv2.calcHist([segima], [i], None, [256], [1, 256])

plt.plot(hist)
plt.show()

#plt.imshow(imaROI1,'Greys')
#plt.show
