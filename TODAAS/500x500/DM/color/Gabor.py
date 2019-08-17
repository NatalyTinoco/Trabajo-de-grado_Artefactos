# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 00:09:34 2019

@author: Nataly
"""

import cv2
import numpy as np
import glob
from skimage.feature import greycomatrix, greycoprops
import skimage.feature
from scipy.stats import kurtosis
import statistics as stats
import pywt
import pywt.data
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
#    plt.imshow(brick)
#    plt.show()
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
#        print(kernel.shape)
        
#        vmin = np.min(powers)
#        vmax = np.max(powers)
        #print(vmin,vmax)
        #print(np.asarray(powers).shape)
    
        for patch in zip(powers):
#            print(np.asarray(patch).shape)
            patch=np.asarray(list(patch))
            #print(type(patch))
#            plt.imshow(patch[0,:,:], vmin=vmin, vmax=vmax)
#            plt.show()
            p[ii]=patch[0,:,:]
            ii=ii+1
    return p       

def laplaciano (V):
    lxx=1/6*np.array([[0,0,1], [0,-2,0],[1,0,0]])
    lyy=1/6*np.array([[1,0,0], [0,-2,0],[0,0,1]])
    
    #tipomas=3
    #Vnu,tama,tamanu=agregarceros(V,tipomas)
    #la=filtro(Vnu,tama,tamanu,tipomas,laplaciano,V)
    lx2=cv2.filter2D(V, -1, lxx)
    ly2=cv2.filter2D(V, -1, lyy)
    lx=np.array([-1,2,-1])
    ly=np.transpose(lx)
    lxd=cv2.filter2D(V, -1, lx)
    lyd=cv2.filter2D(V, -1, ly)
    la=lxd+lyd+lx2+ly2
    print('la',la.shape)
    La=0
    f=la.shape
    suma=0
    for x in range(f[0]):
        for y in range (f[1]):
            suma=suma+abs(la[x,y])
            #print(suma)
    #print(suma)
    f=list(f)
    La=(1/f[0]*f[1])*suma
    varLa=0
    for x in range(f[0]):
        for y in range (f[1]):
            varLa=(varLa)+(la[x,y]-La)**(2)
            #print(suma)
    #print(varLa)
    return La, varLa

mediaDM=[]
medianaDM=[]
desviacionDM=[]
lagaborDM=[]
varlagaborDM=[]
laDM=[]
varlaDM=[]
entropiaDM=[]


  
#from matplotlib import pyplot as plt

for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    aa,bb,c = im.shape    
    croppedrgb=im
    #V=cropped
    cropped=cv2.imread('./V/'+image,0)
    print(cropped.shape)
    p=filtrogabor(cropped)  
    mediaDM.append(np.mean(p[3]))
    medianaDM.append(np.median(p[3])) 
    desviacionDM.append(np.var(p[3]))
    lagaborDM.append(sum(laplaciano (p[3])))
    varlagaborDM.append(sum(laplaciano ( p[3])))
    laDM.append(sum(laplaciano ( croppedrgb)))
    varlaDM.append(sum(laplaciano (croppedrgb)))
    entropiaDM.append(skimage.measure.shannon_entropy(p[3]))
   
import pandas as pd    
datos = {'EntropíaDM':entropiaDM,
         'MediaDM':mediaDM,
         'DesviacionDM':desviacionDM,
         'lAPLACIANOgaborDM':lagaborDM,
         'VarianzalaplacianagaborDM':varlagaborDM,
         'MedianaDM':medianaDM,
         'lAPLACIANODM':laDM,
         'VarianzalaplacianaDM':varlaDM}
datos = pd.DataFrame(datos)
datos.to_excel('GaborDM.xlsx') 
  