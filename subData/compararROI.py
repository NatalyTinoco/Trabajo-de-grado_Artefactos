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
 
def compare_images(imageA, imageB):
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
   #"""

    
import glob

#similitud=np.zeros((307,16))
similitud=np.zeros((307,3))
i=0
for imgfile in glob.glob("*.jpg"):
    #dire='./segROI/#6/Z/'+imgfile
    print(imgfile)  
    #imgfile="WL_00442.jpg"
    original = cv2.imread("./segROI/ROI_Manuales/"+imgfile,0)
    #original=original[:,:,0]
    #plt.imshow(original)
    #plt.show()
    """
    unoB = cv2.imread("./segROI/#1/B/"+imgfile,0)
    unoV = cv2.imread("./segROI/#1/V/"+imgfile,0)
    unoZ = cv2.imread("./segROI/#1/Z/"+imgfile,0)
    dosB = cv2.imread("./segROI/#2/B/"+imgfile,0)
    dosV = cv2.imread("./segROI/#2/V/"+imgfile,0)
    dosZ = cv2.imread("./segROI/#2/Z/"+imgfile,0)                     
    tresB = cv2.imread("./segROI/#3/B/"+imgfile,0)
    tresV = cv2.imread("./segROI/#3/V/"+imgfile,0)
    tresZ = cv2.imread("./segROI/#3/Z/"+imgfile,0)
    cuatroB=cv2.imread("./segROI/#4/B/"+imgfile,0)
    cuatroV=cv2.imread("./segROI/#4/V/"+imgfile,0)
    cuatroZ=cv2.imread("./segROI/#4/Z/"+imgfile,0)
    cincoB=cv2.imread("./segROI/#5/B/"+imgfile,0)
    """
    cincoV=cv2.imread("./segROI/#5/V/"+imgfile,0)
    cincoZ=cv2.imread("./segROI/#5/Z/"+imgfile,0)
    #plt.imshow(original,'Greys')
    #plt.show() 
                        
    s0=ssim(original, original)#, "Original vs. Original"
    """
    s1=ssim(original, unoB)#, "Original vs. unoB"
    s2=ssim(original, unoV)#, "Original vs. unoV"
    s3=ssim(original, unoZ)#, "Original vs. unoZ"
    s4=ssim(original, dosB)#, "Original vs. dosB"
    s5=ssim(original, dosV)#, "Original vs. dosV"
    s6=ssim(original, dosZ)#, "Original vs. dosZ"
    s7=ssim(original, tresB)#, "Original vs. tresB"
    s8=ssim(original, tresV)#, "Original vs. tresV"
    s9=ssim(original, tresV)#, "Original vs. tresZ"
    s10=ssim(original, cuatroB)#, "Original vs. cuatroB"
    s11=ssim(original, cuatroV)#, "Original vs. cuatroV"
    s12=ssim(original, cuatroZ)#, "Original vs. cuatroZ"
    s13=ssim(original, cincoB, )#"Original vs. cincoB"
    """
    s14=ssim(original, cincoV)#, "Original vs. cincoV"
    s15=ssim(original, cincoZ)#, "Original vs. cincoZ"
    
    similitud[i,0]=s0
    """
    similitud[i,1]=s1
    similitud[i,2]=s2
    similitud[i,3]=s3
    similitud[i,4]=s4
    similitud[i,5]=s5
    similitud[i,6]=s6
    similitud[i,7]=s7
    similitud[i,8]=s8
    similitud[i,9]=s9
    similitud[i,10]=s10
    similitud[i,11]=s11
    similitud[i,12]=s12
    similitud[i,13]=s13
    """
    similitud[i,1]=s14
    similitud[i,2]=s15
      
    i=i+1
    
import openpyxl

doc = openpyxl.load_workbook('ROI.xlsx')
doc.get_sheet_names()
hoja = doc.get_sheet_by_name('Hoja1')
#table = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
table = ['A','B','C']

i=0
ii=0
for a in range (int(len(table))):
    #print(i)
    #print(a)
    for x in range ((len(similitud[:,1]))):
        hoja[table[i]+ str (x+4)]=similitud[x,a]
    #print(table[i])
    #print(table[i+1])
    #print(table[i+2])
    print(x,a)
    i=(a+1)
    #print(a)
doc.save("ROI.xlsx")