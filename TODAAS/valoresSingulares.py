# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:26:02 2019

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

def SingularValueFeature(A):
    #import numpy.linalg as svd 
    k,k1=A.shape
    U,s,V=np.linalg.svd(A,full_matrices=False)
    #print(U.shape,s.shape,V.shape)
    reconst_matrix=np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return  reconst_matrix,s
        
beta=[]
sumas=[]
media=[]
mediana=[]
destan=[]
var=[]
correla=[]
covar=[]      

betaRE=[]
sumasRE=[]
mediaRE=[]
medianaRE=[]
destanRE=[]
varRE=[]
correlaRE=[]
covarRE=[]   

betaNO=[]
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
                    B,T=SingularValueFeature(cropped)
                    T=T.tolist() 
                    u=np.max(T)
                    TT= T.index(u)
                    if TT==0:
                       betadm=T[TT]/sum(T)
                    else:
                        betadm=sum(T[0:TT])/sum(T)
                    beta.append(betadm)
                    sumas.append(sum(T))
                    media.append(np.mean(T))
                    mediana.append(np.median(T))
                    destan.append(np.std(T))
                    var.append(np.var(T))
                    correla.append(np.corrcoef(T))
                    covar.append(np.cov(T))                
                   

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
                    BRE,TRE=SingularValueFeature(cropped2)
                    TRE=TRE.tolist() 
                    uRE=np.max(TRE)
                    TTRE= TRE.index(uRE)
                    if TTRE==0:
                       betaREe=TRE[TTRE]/sum(TRE)
                    else:
                        betaREe=sum(TRE[0:TTRE])/sum(TRE)
                    betaRE.append(betaREe)
                    sumasRE.append(sum(TRE))
                    mediaRE.append(np.mean(TRE))
                    medianaRE.append(np.median(TRE))
                    destanRE.append(np.std(TRE))
                    varRE.append(np.var(TRE))
                    correlaRE.append(np.corrcoef(TRE))
                    covarRE.append(np.cov(TRE))  
            
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
                    BNO,TNO=SingularValueFeature(cropped3)
                    TNO=TNO.tolist() 
                    uNO=np.max(TNO)
                    TTNO= TNO.index(uNO)
                    if TTNO==0:
                       betaNOo=TNO[TTNO]/sum(TNO)
                    else:
                        betaNOo=sum(TNO[0:TTNO])/sum(TNO)
                    betaNO.append(betaNOo)
                    sumasNO.append(sum(TNO))
                    mediaNO.append(np.mean(TNO))
                    medianaNO.append(np.median(TNO))
                    destanNO.append(np.std(TNO))
                    varNO.append(np.var(TNO))
                    correlaNO.append(np.corrcoef(TNO))
                    covarNO.append(np.cov(TNO))        
         
           
import pandas as pd    
datos = {'beta':beta,
         'sumas':sumas,
         'media':media,
         'mediana':mediana,
         'desviacion E':destan,
         'Varianza':var,
         'Corralación':correla,
         'Covarianza':covar}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasValoresSingulares_DM.xlsx')             

datos = {'betaRE':betaRE,
         'sumasRE':sumasRE,
         'mediaRE':mediaRE,
         'medianaRE':medianaRE,
         'desviacion ERE':destanRE,
         'VarianzaRE':varRE,
         'CorralaciónRE':correlaRE,
         'CovarianzaRE':covarRE}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasValoresSingulares_RE.xlsx')  

datos = {'betaNO':betaNO,
         'sumasNO':sumasNO,
         'mediaNO':mediaNO,
         'medianaNO':medianaNO,
         'desviacion ENO':destanNO,
         'VarianzaNO':varNO,
         'CorralaciónNO':correlaNO,
         'CovarianzaNO':covarNO}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasValoresSingulares_NO.xlsx')  

           
