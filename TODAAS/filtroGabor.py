# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:45:59 2019

@author: Nataly
"""

import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt
from rOI import ROI
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from scipy.stats import kurtosis
import statistics as stats
import pywt
import pywt.data
#from __future__ import print_function
from scipy import ndimage as nd
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


#tamañoA = []
#tamañoB = []
def Fourier(inA):
    f = np.fft.fft2(inA)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    return fourier 
    
def GLCM (imA):
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
def tama(a,b):
    if a<600 or b<600:
        tamañoA = 200
        tamañoB = 200
    else:
        tamañoA = 600
        tamañoB = 600
    return tamañoA,tamañoB


def filtrogabor(croppedimg):
    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = nd.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats
    
    def match(feats, ref_feats):
        min_error = np.inf
        min_i = None
        for i in range(ref_feats.shape[0]):
            error = np.sum((feats - ref_feats[i, :])**2)
            if error < min_error:
                min_error = error
                min_i = i
        return min_i
    
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    
    shrink = (slice(0, None, 3), slice(0, None, 3))
    #brick = img_as_float(data.load('brick.png'))[shrink]
    brick = img_as_float(croppedimg)[shrink]
    #grass = img_as_float(data.load('grass.png'))[shrink]
    #wall = img_as_float(data.load('rough-wall.png'))[shrink]
    #image_names = ('brick')
    #images = (brick)
    plt.imshow(brick)
    plt.show()
    img=brick.copy()
    # prepare reference features
    ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
    ref_feats[0, :, :] = compute_feats(brick, kernels)
    #ref_feats[1, :, :] = compute_feats(grass, kernels)
    #ref_feats[2, :, :] = compute_feats(wall, kernels)
    
    #print('Rotated images matched against references using Gabor filter banks:')
    
    #print('original: brick, rotated: 30deg, match result: ', end='')
    #feats = compute_feats(nd.rotate(brick, angle=190, reshape=False), kernels)
    #print(image_names[match(feats, ref_feats)])
    
    #print('original: brick, rotated: 70deg, match result: ', end='')
    #feats = compute_feats(nd.rotate(brick, angle=70, reshape=False), kernels)
    #print(image_names[match(feats, ref_feats)])
    
    #print('original: grass, rotated: 145deg, match result: ', end='')
    #feats = compute_feats(nd.rotate(grass, angle=145, reshape=False), kernels)
    #print(image_names[match(feats, ref_feats)])
    
    
    def power(image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
                       nd.convolve(image, np.imag(kernel), mode='wrap')**2)
    
    # Plot a selection of the filter bank kernels and their responses.
    results = []
    kernel_params = []
    for theta in (0, 1):
        theta = theta / 4. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, [power(img, kernel)]))
#                    plt.imshow(img)
#                    plt.show()
    
    #for label, (kernel, powers) in zip(kernel_params, results):
    j1=0
    j2=0
    j3=0
    j4=0
    j=[j1,j2,j3,j4]
    p1=0
    p2=0
    p3=0
    p4=0
    p=[p1,p2,p3,p4]
    i=0
    ii=0
    for label, (kernel, powers) in zip(kernel_params, results):
#                        plt.imshow(np.real(kernel), interpolation='nearest')
#                        plt.show()
        j[i]=kernel
        i=i+1
        print(kernel.shape)
        
        vmin = np.min(powers)
        vmax = np.max(powers)
        #print(vmin,vmax)
        #print(np.asarray(powers).shape)
    
        for patch in zip(powers):
            print(np.asarray(patch).shape)
            patch=np.asarray(list(patch))
            #print(type(patch))
            plt.imshow(patch[0,:,:], vmin=vmin, vmax=vmax)
            plt.show()
            p[ii]=patch[0,:,:]
            ii=ii+1
    return[]        
    
    
for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    #cv2.imshow('Grays',imaROI)
    #cv2.destroyAllWindows()
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
    V=V*imaROI
        
    for z in range(c):
        im[:,:,z]=im[:,:,z]*imaROI
    
    
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    filetxt=image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    dm=0
    imunda=0
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            print('========================================DM========================================')
            
            dm=dm+1   
            #print(image,dm)
            a,b= V[int(y1):int(y2),int(x1):int(x2)].shape
            tamañoA,tamañoB=tama(a,b)
            V1= V[int(y1):int(y2),int(x1):int(x2)]
            vecesA = int(a/tamañoA)
            vecesB = int(b/tamañoB)
        
            for f in range(0,a-tamañoA,tamañoA):
                for c in range(0,b-tamañoB,tamañoB):
                    #print(f,c)
                    cropped = V1[f:f+tamañoA,c:c+tamañoB]
                    croppedrgb = im[f:f+tamañoA,c:c+tamañoB]
                   
                    #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
                    if c==tamañoB*vecesB-tamañoB:
                        cropped = V1[f:f+tamañoA,c:]
                        croppedrgb = im[f:f+tamañoA,c:]
                        #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
                    if f==tamañoA*vecesA-tamañoA:
                         #print('ola')
                         if c==tamañoB*vecesB-tamañoB:
                            cropped = V1[f:,c:]
                            croppedrgb = im[f:,c:]
                       
                             #test2[f:,c:]=test[f:,c:]
                         else:
                             cropped = V1[f:,c:c+tamañoB]
                             croppedrgb = im[f:,c:c+tamañoB]
                    
                    filtrogabor(cropped)  
                             #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                             #print('dani')
                    #cropFou=cropped
#                    cropped_1=cropped.copy()
#                    croppedrgb_1=croppedrgb.copy()
                    
                    
        if cls==0:
            re=re+1        
            print(re)
        if cls==2:
            imunda=imunda+1
#        imSinBBOX[int(y1):int(y2),int(x1):int(x2)]=0
        
#        print('cls', cls)
#        if cls!=0 and cls!=1 and cls!=2 and cls!=3 and cls!=4 and cls!=5 and cls!=6:
#             plt.imshow(im)
#             plt.show() 
#            re=re+1
    if re > 0 and dm==0 and imunda==0:
        print('========================================RE========================================')
            
        inta=V[y3:y3+h3,x3:x3+w3]
        aa,bb=inta.shape
        tamañoA,tamañoB=tama(aa,bb)
        vecesA = int(aa/tamañoA)
        vecesB = int(bb/tamañoB)
        
        for f in range(0,aa-tamañoA,tamañoA):
            for c in range(0,bb-tamañoB,tamañoB):
                cropped2 = inta[f:f+tamañoA,c:c+tamañoB]
                croppedrgb2 = im[f:f+tamañoA,c:c+tamañoB]
                if c==tamañoB*vecesB-tamañoB:
                    cropped2 = inta[f:f+tamañoA,c:]
                    croppedrgb2 = im[f:f+tamañoA,c:]
                if f==tamañoA*vecesA-tamañoA:
                     if c==tamañoB*vecesB-tamañoB:
                        cropped2 = inta[f:,c:]
                        croppedrgb2 = im[f:,c:]
                     else:
                         cropped2 = inta[f:,c:c+tamañoB]
                         croppedrgb2 = im[f:,c:c+tamañoB]
                cropped2_1=cropped2.copy()
                croppedrgb2_1=croppedrgb2.copy()    
                filtrogabor(cropped2)  
    if re==0 and dm==0 and imunda==0:
        print('========================================NO========================================')
            
        inta3=V[y3:y3+h3,x3:x3+w3]
        aaa,bbb=inta3.shape
        tamañoA,tamañoB=tama(aaa,bbb)
        vecesA = int(aaa/tamañoA)
        vecesB = int(bbb/tamañoB)
        
        for f in range(0,aaa-tamañoA,tamañoA):
            for c in range(0,bbb-tamañoB,tamañoB):
                cropped3 = inta3[f:f+tamañoA,c:c+tamañoB]
                croppedrgb3 = im[f:f+tamañoA,c:c+tamañoB]
                if c==tamañoB*vecesB-tamañoB:
                    cropped3 = inta3[f:f+tamañoA,c:]
                    croppedrgb3 = im[f:f+tamañoA,c:]
                if f==tamañoA*vecesA-tamañoA:
                     if c==tamañoB*vecesB-tamañoB:
                        cropped3 = inta3[f:,c:]
                        croppedrgb3 = im[f:,c:]
                     else:
                         cropped3 = inta3[f:,c:c+tamañoB]
                         croppedrgb3 = im[f:,c:c+tamañoB]
                cropped3_1=cropped3.copy()
                croppedrgb3_1=croppedrgb3.copy()            
                filtrogabor(cropped3)  



