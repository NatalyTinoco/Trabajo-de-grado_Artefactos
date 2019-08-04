# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:21:24 2019

@author: Usuario
"""

import cv2
from matplotlib import pyplot as plt
from readimg import read_img

img = read_img('00000.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.GaussianBlur(img, (3, 3), 0)

edges = cv2.Laplacian(img, -1, ksize=27, scale=1, borderType = cv2.BORDER_DEFAULT)

output = [img,edges]
titles = ['original','edges']

for i in range(2):
    plt.subplot(1,2, i+1)
    plt.imshow(output[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

#%%

img = read_img('00000.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.GaussianBlur(img, (3, 3), 0)

edgesx = cv2.Sobel(img, -1, dx=5, dy=0, ksize=7
                   , scale=1,
                   delta=0, borderType = cv2.BORDER_DEFAULT)

edgesy = cv2.Sobel(img, -1, dx=0, dy=5, ksize=7, scale=1,
                   delta=0, borderType = cv2.BORDER_DEFAULT)

edges = edgesx + edgesy

output = [img,edgesx,edgesy,edges]
titles = ['original','dx=1 dy=0','dx=0 dy=1','edges']

for i in range(4):
    plt.subplot(2,2, i+1)
    plt.imshow(output[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

#%%

img = read_img('00000.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.GaussianBlur(img, (3, 3), 0)

edgesx = cv2.Scharr(img, -1, dx=1, dy=0, scale=1,
                   delta=0, borderType = cv2.BORDER_DEFAULT)

edgesy = cv2.Scharr(img, -1, dx=0, dy=1, scale=1,
                   delta=0, borderType = cv2.BORDER_DEFAULT)

edges = edgesx + edgesy

output = [img,edgesx,edgesy,edges]
titles = ['original','dx=1 dy=0','dx=0 dy=1','edges']

for i in range(4):
    plt.subplot(2,2, i+1)
    plt.imshow(output[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()