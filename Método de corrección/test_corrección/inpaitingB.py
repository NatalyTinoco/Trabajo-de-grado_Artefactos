# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:36:36 2019

@author: Nataly
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.restoration import inpaint
import cv2

file='00033'
image_orig=cv2.imread(file+'.jpg')
#image_orig=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
##plt.imshow(image_orig)
##plt.show()
mask =cv2.imread(file+'_seg.jpg',0)

#cv2.imshow('',mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Defect image over the same region in each color channel
image_defect = image_orig.copy()
for layer in range(image_defect.shape[-1]):
    image_defect[np.where(mask)] = 0

#plt.imshow(image_defect)
#plt.show()
image_result =inpaint.inpaint_biharmonic(image_orig, mask,multichannel=True)

image_result =inpaint.inpaint_biharmonic(image_defect, mask, multichannel=3)
#plt.imshow(image_result)
#plt.show()

#cv2.imshow('',image_result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#

imgs= cv2.normalize(image_result, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
cv2.imshow('',imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()


dire='./'+file+'_inpaiting_B.png'
cv2.imwrite(dire,imgs)
k = cv2.waitKey(1000)
cv2.destroyAllWindows()

