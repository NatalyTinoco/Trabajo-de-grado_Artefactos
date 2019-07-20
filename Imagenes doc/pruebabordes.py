# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 21:52:39 2019

@author: Nataly
"""

from readimg import read_img #leer imagines ### img=read_img(imgfile)##
import cv2
import pylab as plt 
from matplotlib import pyplot as plt
from scipy import stats # importando scipy.stats
import numpy as np
import glob
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
from scipy.spatial import distance
import statistics

from equalization import globalequalization, adaptativeequalization, contraststretching
from Filtros import filtroautolevel, filtromediana, filtrominimo, filtrogaussiano,filtropromedio

from skimage import feature
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h


medidas=np.zeros((166,3))
i=0
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

for file in glob.glob("*.jpg"):

    imgfile=file
    filetxt=file[0:len(file)-3]+'txt'      
    bboxfile=filetxt
    img = read_img(imgfile)
    #img1=filtrominimo(img)
    #img=adaptativeequalization(img)
    #img=contraststretching(img)
    imgg=img.copy()
    img1=img
    img2 = cv2.Canny(img1,54,90)   
    #img2=auto_canny(img1,sigma=0.2)
    #img2 = roberts(img1)
    #img2= sobel(image)
    imgg=cv2.cvtColor(imgg,cv2.COLOR_RGB2BGR)
    img1=cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    fig, ax = plt.subplots(ncols=3, figsize=(13,7), sharex=True, sharey=True)
    ax[0].imshow(imgg)
    ax[0].set_title('Log')
    ax[1].imshow(img1)
    ax[1].set_title('Filtrada')
    ax[2].imshow(img2,cmap = 'gray')
    ax[2].set_title('Borde')
    plt.show()
    
    fig, ax = plt.subplots(ncols=3, figsize=(13,7), sharex=True, sharey=True)
    ax[0].hist(imgg.ravel(),255,[1,imgg.max()])
    ax[1].hist(img1.ravel(),256,[1,img1.max()])
    ax[2].hist(img2.ravel(),255,[1,img2.max()])
    plt.show()
    i=i+1
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()