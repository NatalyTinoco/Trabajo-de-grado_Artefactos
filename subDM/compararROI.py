# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:27:13 2019

@author: Nataly
"""
from skimage.measure import compare_ssim as ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()
    
import glob

for imgfile in glob.glob("*.jpg"):
    #dire='./segROI/#6/Z/'+imgfile
    original = cv2.imread("./segROI/ROI_Manuales/"+imgfile,0)
    uno = cv2.imread("./segROI/#1/"+imgfile,0)
    dos = cv2.imread("./segROI/#2/"+imgfile,0)
    tres = cv2.imread("./segROI/#3/"+imgfile,0)
    cuatro=cv2.imread("./segROI/#4/"+imgfile,0)
    cincoB=cv2.imread("./segROI/#5/B/"+imgfile,0)
    cincoV=cv2.imread("./segROI/#5/V/"+imgfile,0)
    cincoZ=cv2.imread("./segROI/#5/Z/"+imgfile,0)
    #plt.imshow(original,'Greys')
    #plt.show() 
                        
    compare_images(original, original, "Original vs. Original")
    compare_images(original, uno, "Original vs. uno")
    compare_images(original, dos, "Original vs. dos")
    compare_images(original, tres, "Original vs. tres")
    compare_images(original, cuatro, "Original vs. cuatro")
    compare_images(original, cincoB, "Original vs. cincoB")
    compare_images(original, cincoV, "Original vs. cincoV")
    compare_images(original, cincoZ, "Original vs. cincoZ")
    
    
