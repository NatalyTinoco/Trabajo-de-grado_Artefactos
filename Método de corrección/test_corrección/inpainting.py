# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:53:06 2019

@author: Daniela
"""

import cv2
import pylab as plt

file='00000'
ima=cv2.imread(file+'.jpg')
mask =cv2.imread(file+'_seg.jpg',0)
mask= cv2.normalize(mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

size = ima.shape


ns = cv2.inpaint(ima,mask, 3, cv2.INPAINT_NS)
#cv2.imshow("ns", ns)
#cv2.waitKey(0)

dire='./'+file+'_inpaitingNs.jpg'
cv2.imwrite(dire,ns)
k = cv2.waitKey(1000)
cv2.destroyAllWindows()

telea = cv2.inpaint(ima,mask, 3, cv2.INPAINT_TELEA)
#cv2.imshow("telea", telea)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

dire='./'+file+'_inpaitingTELEA.jpg'
cv2.imwrite(dire,telea)
k = cv2.waitKey(1000)
cv2.destroyAllWindows()




