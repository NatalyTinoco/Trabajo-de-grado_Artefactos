# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:24:21 2019

@author: Nataly
"""
import cv2 
import glob 
import sys
import numpy as np
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from yolovoc import yolo2voc
from readboxes import read_boxes
from rOI import ROI
from ventaneo import ventaneoo

from matplotlib import pyplot as plt


def Fourier(inA):
    f = np.fft.fft2(inA)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    return fourier 
    
def GLCM (imA):    
        from skimage.feature import greycomatrix, greycoprops
        import skimage.feature
        a=int(np.max(imA))
        g = skimage.feature.greycomatrix(imA, [1], [0], levels=a+1, symmetric=False, normed=True)                  
        contraste=skimage.feature.greycoprops(g, 'contrast')[0][0]
        energia=skimage.feature.greycoprops(g, 'energy')[0][0]
        homogeneidad=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
        correlacion=skimage.feature.greycoprops(g, 'correlation')[0][0]
        disimi= greycoprops(g, 'dissimilarity') 
        ASM= greycoprops(g, 'ASM')
        entropia=skimage.measure.shannon_entropy(g) 
        return g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia
#                    plt.imshow(cropped)

def get_blur_degree(img, sv_num=10):
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv


def get_blur_map(img, win_size=10, sv_num=3):
    #img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    new_img = np.zeros((img.shape[0]+win_size*2, img.shape[1]+win_size*2))
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if i<win_size:
                p = win_size-i
            elif i>img.shape[0]+win_size-1:
                p = img.shape[0]*2-i
            else:
                p = i-win_size
            if j<win_size:
                q = win_size-j
            elif j>img.shape[1]+win_size-1:
                q = img.shape[1]*2-j
            else:
                q = j-win_size
#            print (p,q, i, j)
            new_img[i,j] = img[p,q]
    blur_map = np.zeros((img.shape[0], img.shape[1]))
    max_sv = 0
    min_sv = 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            block = new_img[i:i+win_size*2, j:j+win_size*2]
            u, s, v = np.linalg.svd(block)
            top_sv = np.sum(s[0:sv_num])
            total_sv = np.sum(s)
            sv_degree = top_sv/total_sv
            if max_sv < sv_degree:
                max_sv = sv_degree
            if min_sv > sv_degree:
                min_sv = sv_degree
            blur_map[i, j] = sv_degree
    blur_map = (blur_map-min_sv)/(max_sv-min_sv)
    return blur_map
def DFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape 
    crow,ccol = int(rows/2) , int(cols/2)
    print(rows,cols,crow,ccol)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back



#import radialProfile
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

import pylab as py
def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]

import math
from scipy.stats import linregress

for imgfile in glob.glob("*.jpg"):
#    imgfile='00070_batch2.jpg'
#    imgfile='00064_batch2.jpg'
#    imgfile='00272.jpg'
   
    img=cv2.imread(imgfile)  
    imaROI=ROI(img)
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    imgcop=img.copy()
#    plt.imshow(img)
#    plt.show()
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range(3):
        img[:,:,z]=img[:,:,z]*imaROI
              
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    
    HSV=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
    V=V[y3:y3+h3,x3:x3+w3]

    img_back=DFT(V)
    j=azimuthalAverage(img_back)
    xs=np.arange(len(j))
    slope = linregress(xs, j)[0]  # slope in units of y / x
    slope_angle = math.atan(slope)  # slope angle in radians
    alfa0 = math.degrees(slope_angle) 
    
    gradiente=np.gradient(j)
    alfa0ma=np.max(gradiente)               
    a,b=V.shape
    tamañoa1A=150
    tamañoa1B=150
    rea1=0
    Binary=S.copy()
    for fa1 in range(0,a,tamañoa1A):
       for ca1 in range(0,b,tamañoa1B):
            croppeda1=ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, V)
            vecesA = int(a/tamañoa1A)
            vecesB = int(b/tamañoa1B)
            img_backc=DFT(croppeda1)
            jc=azimuthalAverage(img_backc)
            gradientec=np.gradient(jc)
            alfapma=np.max(gradientec)
            xsc=np.arange(len(jc))
            slopec = linregress(xsc, jc)[0]  # slope in units of y / x
            slope_anglec = math.atan(slopec)  # slope angle in radians
            alfap = math.degrees(slope_anglec) 
            print('ALFA',alfap)
            ta1,ta2=croppeda1.shape
#            plt.imshow(croppeda1)
#            plt.show()
            print(fa1,ca1)
#            binary=croppeda1.copy()
#            if alfap > alfa0:
            if  alfapma> alfa0ma:
                #if s[f,c]<h[f,c]:
                binary=1
            else:
                binary=0
            Binary[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B]=binary
            #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
            if ca1==tamañoa1B*vecesB-tamañoa1B:
                #cropped = V[f:f+tamañoA,c:]
                Binary[fa1:fa1+tamañoa1A,ca1:]=binary
                #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
            if fa1==tamañoa1A*vecesA-tamañoa1A:
                 if ca1==tamañoa1B*vecesB-tamañoa1B:
#                    cropped = V[f:,c:]
                    Binary[fa1:,ca1:]=binary
                 else:
                     Binary[fa1:,ca1:ca1+tamañoa1B]=binary
       
            if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:
               if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
                     Binary[fa1:a,ca1:b]=binary
               else:
                      Binary[fa1:fa1+tamañoa1A,ca1:b]=binary
            if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
                 print('ola')
                 if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:

                     Binary[fa1:a,ca1:b]=binary
                     
                 else:
                     Binary[fa1:b,ca1:ca1+tamañoa1B]=binary
            rea1=rea1+1
    fila,Col,shu=imgcop.shape
    Binaryfinal=np.zeros((fila,Col)).astype(np.uint8)
    if Binary.shape==Binaryfinal.shape:
        Binaryfinal=Binary
    else:
        Binaryfinal[y3:y3+h3,x3:x3+w3]=Binary   
    Binaryfinal=Binaryfinal*imaROI
    Binaryfinal=Binaryfinal*255
    dire='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/subDM/umb_DM_1.2/'+imgfile
    cv2.imwrite(dire,Binaryfinal)
#    cv2.imshow('image',Binaryfinal)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
    