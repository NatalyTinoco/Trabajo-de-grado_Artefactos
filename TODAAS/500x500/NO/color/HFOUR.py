# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 01:18:58 2019

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

def HOP(image):
    import matplotlib.pyplot as plt
    from skimage.feature import hog
    from skimage import data, exposure
    #image = data.astronaut()
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled  
#def variance_of_laplacian(image):
#	 varl=cv2.Laplacian(image, cv2.CV_64F).var()
#    return varl

contrast=[]
energi=[]
homogenei=[]
correlaci=[]
disi=[]
AS=[]
entrop=[]


pico=[]
sumas=[]
media=[]
mediana=[]
destan=[]
var=[]
    
 
#from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    aa,bb,c = im.shape    
    croppedrgb=im
    croppedrgb_2=croppedrgb.copy()
    #V=cropped
    cropped=cv2.imread('./V/'+image,0)
    cropped=Fourier(cropped)
    hist = cv2.calcHist([cropped],[0],None,[256],[0,255])
#                    plt.plot(hist)
#                    plt.show()
    hisa=hist.copy()
    hist=hist.tolist() 
    u=np.max(hist)
    hi=hist.index(u)
    pico.append(hi)
    sumas.append(sum(hisa))
    media.append(np.mean(hisa))
    mediana.append(np.median(hisa))
    destan.append(np.std(hisa))
    var.append(np.var(hisa))



   
import pandas as pd    
datos = {'Pico':pico,
         'sumas':sumas,
         'media':media,
         'mediana':mediana,
         'desviacion E':destan,
         'Varianza':var}
datos = pd.DataFrame(datos)
datos.to_excel('HFOURNO.xlsx') 