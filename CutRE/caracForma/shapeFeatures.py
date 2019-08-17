# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:04:25 2019

@author: Usuario
"""

import cv2
from readboxes import read_boxes
from yolovoc import yolo2voc
from cutboxes import cut_boxes
import glob
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('00000_seg.jpg')
#im = cv2.imread('00000.jpg')
boxes = read_boxes('00000.txt')
boxes_abs = yolo2voc(boxes, im.shape)
re=0
dm=0

idn = cut_boxes(boxes_abs,im,re,dm)

for image in glob.glob('segme/*.jpg'):
    img = cv2.imread('bboxreflejo (7).jpg',0)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
    cnt = contours[0]
    M = cv2.moments(cnt)
    
    #centroid
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    area = cv2.contourArea(cnt)
    
    #contour
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    
    #convex
    hull = cv2.convexHull(cnt,False)
    k = cv2.isContourConvex(cnt)
    
    x,y,w,h = cv2.boundingRect(cnt)
    img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img3 = cv2.drawContours(img,[box],0,(0,0,255),2)
    
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    img4 = cv2.circle(img,center,radius,(0,255,0),2)
    
    cv2.imshow('image',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('okk')
    
#%%
    
image_seg = cv2.imread('00000_seg.jpg',0)
im_rgb = image_seg.copy()
im_rgb = cv2.cvtColor(image_seg, cv2.COLOR_BGR2RGB)
plt.imshow(im_rgb)

gray = cv2.cvtColor(src = image_seg, code = cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(src = image_seg, 
    ksize = (5, 5), 
    sigmaX = 0)

(t, binary) = cv2.threshold(src = blur,
    thresh = 245, 
    maxval = 255, 
    type = cv2.THRESH_BINARY)

(_, contours) = cv2.findContours(image = image_seg, 
    mode = cv2.RETR_EXTERNAL,
    method = cv2.CHAIN_APPROX_SIMPLE)

ret,thresh = cv2.threshold(image_seg,245,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))
    
ctr = np.array(contours).reshape((-1,1,2)).astype(np.int32)

a = cv2.drawContours(image = image_seg, 
                     contours = contours, 
                     contourIdx = -1, 
                     color = (0, 0, 255), 
                     thickness = 5)

cnt = contours[4]
cv2.drawContours(image_seg, [cnt], 0, (0,255,0), 3)


mask = np.zeros(shape = image_seg.shape, dtype = "uint8")

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(img = mask, 
        pt1 = (x, y), 
        pt2 = (x + w, y + h), 
        color = (255, 255, 255), 
        thickness = -1)

image = cv2.bitwise_and(src1 = image_seg, src2 = mask)

plt.hist(binary.ravel(),256,[0,256])
plt.show()

pp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(pp)

cv2.imshow('image',a)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
