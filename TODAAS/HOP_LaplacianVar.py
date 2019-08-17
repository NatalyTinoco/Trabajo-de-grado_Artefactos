# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:09:01 2019

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
  
varla=[]
sumas=[]
media=[]
mediana=[]
destan=[]
var=[]
correla=[]
covar=[]      

varlaRE=[]
sumasRE=[]
mediaRE=[]
medianaRE=[]
destanRE=[]
varRE=[]
correlaRE=[]
covarRE=[]   

varlaNO=[]
sumasNO=[]
mediaNO=[]
medianaNO=[]
destanNO=[]
varNO=[]
correlaNO=[]
covarNO=[]   


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
    imSinBBOX=im.copy()
    tamañoA = 500
    tamañoB = 500        
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            print('DM')
            
            dm=dm+1   
            #print(image,dm)
            a,b= V[int(y1):int(y2),int(x1):int(x2)].shape
            #tamañoA,tamañoB=tama(a,b)
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
                       
                             #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                             #print('dani')
                    cropped=cv2.resize(cropped,(500,500))      
                    croppedrgb=cv2.resize(croppedrgb,(500,500))  
                    croppedrgb_2=croppedrgb.copy()
                    gradiente=HOP(croppedrgb)
                    varla.append(cv2.Laplacian(croppedrgb_2, cv2.CV_64F).var())
                    sumas.append(sum(gradiente))
                    media.append(np.mean(gradiente))
                    mediana.append(np.median(gradiente))
                    destan.append(np.std(gradiente))
                    var.append(np.var(gradiente))
                    correla.append(np.corrcoef(gradiente))
                    covar.append(np.cov(gradiente))                
                   

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
            inta=V[y3:y3+h3,x3:x3+w3]
            aa,bb=inta.shape
#            tamañoA,tamañoB=tama(aa,bb)
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
                    cropped2=cv2.resize(cropped2,(500,500))      
                    croppedrgb2=cv2.resize(croppedrgb2,(500,500))  
                    croppedrgb2_2=croppedrgb2.copy()
                    gradiente2=HOP(croppedrgb2)
                    varlaRE.append(cv2.Laplacian(croppedrgb2_2, cv2.CV_64F).var())
                    sumasRE.append(sum(gradiente2))
                    mediaRE.append(np.mean(gradiente2))
                    medianaRE.append(np.median(gradiente2))
                    destanRE.append(np.std(gradiente2))
                    varRE.append(np.var(gradiente2))
                    correlaRE.append(np.corrcoef(gradiente2))
                    covarRE.append(np.cov(gradiente2))  
            
    if re==0 and dm==0 and imunda==0:
            inta3=V[y3:y3+h3,x3:x3+w3]
            aaa,bbb=inta3.shape
#            tamañoA,tamañoB=tama(aaa,bbb)
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
                    cropped3=cv2.resize(cropped3,(500,500))      
                    croppedrgb3=cv2.resize(croppedrgb3,(500,500))  
                    croppedrgb3_2=croppedrgb3.copy()
                    gradiente3=HOP(croppedrgb3)
                    varlaNO.append(cv2.Laplacian(croppedrgb3_2, cv2.CV_64F).var())
                    sumasNO.append(sum(gradiente3))
                    mediaNO.append(np.mean(gradiente3))
                    medianaNO.append(np.median(gradiente3))
                    destanNO.append(np.std(gradiente3))
                    varNO.append(np.var(gradiente3))
                    correlaNO.append(np.corrcoef(gradiente3))
                    covarNO.append(np.cov(gradiente3))        
         
           
import pandas as pd    
datos = {'Varianza Laplaciana':varla,
         'sumas':sumas,
         'media':media,
         'mediana':mediana,
         'desviacion E':destan,
         'Varianza':var,
         'Corralación':correla,
         'Covarianza':covar}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasHOP_VarLa_DM.xlsx')             

datos = {'Varianza LaplacianaRE':varlaRE,
         'sumasRE':sumasRE,
         'mediaRE':mediaRE,
         'medianaRE':medianaRE,
         'desviacion ERE':destanRE,
         'VarianzaRE':varRE,
         'CorralaciónRE':correlaRE,
         'CovarianzaRE':covarRE}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasHOP_VarLa_RE.xlsx')  

datos = {'Varianza LaplacianaNO':varlaNO,
         'sumasNO':sumasNO,
         'mediaNO':mediaNO,
         'medianaNO':medianaNO,
         'desviacion ENO':destanNO,
         'VarianzaNO':varNO,
         'CorralaciónNO':correlaNO,
         'CovarianzaNO':covarNO}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasHOP_VarLa_NO.xlsx')  



           