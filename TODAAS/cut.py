# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:56:35 2019

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
        return contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia
#                    plt.imshow(cropped)
    
contrast=[]
energi=[]
homogenei=[]
correlaci=[]
disi=[]
AS=[]
entrop=[]

contrastRE=[]
energiRE=[]
homogeneiRE=[]
correlaciRE=[]
disiRE=[]
ASRE=[]
entropRE=[]

contrastNO=[]
energiNO=[]
homogeneiNO=[]
correlaciNO=[]
disiNO=[]
ASNO=[]
entropNO=[]


for image in glob.glob('*.jpg'):
    #image = '00004_batch2.jpg'
    im = cv2.imread(image)    
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    #cv2.imshow('Grays',imaROI)
    #cv2.destroyAllWindows()
    
#    YUV=cv2.cvtColor(im,cv2.COLOR_RGB2YUV)
#    Y,U,V=cv2.split(YUV)
#    V=V*imaROI
#        
    for z in range(c):
        im[:,:,z]=im[:,:,z]*imaROI
    V=im
    
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    #""" 
#    cv2.imshow("Show",im[y:y+h,x:x+w])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    imf=im.copy()
#    cv2.rectangle(imf,(x,y),(x+w,y+h),(0,255,0),2)
#    cv2.imshow("Show",imf)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    #"""
    #plt.imshow(im)
    #plt.show()
    #imagenROI=im*imaROI
    filetxt=image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    dm=0
    imunda=0
    imSinBBOX=im.copy()
    tamañoA = 200
    tamañoB = 200
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            print('DM')
            
            dm=dm+1   
            #print(image,dm)
            a,b,ch= V[int(y1):int(y2),int(x1):int(x2)].shape
            V1=V[int(y1):int(y2),int(x1):int(x2)]
            #print(a,b)
#            plt.imshow(V[int(y1):int(y2),int(x1):int(x2)])
#            plt.show() 
            vecesA = int(a/tamañoA)
            vecesB = int(b/tamañoB)
        
            for f in range(0,a-tamañoA,tamañoA):
                for c in range(0,b-tamañoB,tamañoB):
                    #print(f,c)
                    cropped = V1[f:f+tamañoA,c:c+tamañoB]
                   
                    #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
                    if c==tamañoB*vecesB-tamañoB:
                        cropped = V1[f:f+tamañoA,c:]
                   
                        #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
                    if f==tamañoA*vecesA-tamañoA:
                         #print('ola')
                         if c==tamañoB*vecesB-tamañoB:
                            cropped = V1[f:,c:]
                   
                             #test2[f:,c:]=test[f:,c:]
                         else:
                             cropped = V1[f:,c:c+tamañoB]
                             #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                             #print('dani')
                
                    dire='./DM_ven200x200/'+str(c)+str(f)+'-' +image 
                    cv2.imwrite(dire,cropped)
                    
        if cls==0:
            re=re+1        
            print(re)
        if cls==2:
            imunda=imunda+1
            
        imSinBBOX[int(y1):int(y2),int(x1):int(x2)]=0
        dire='./sinArtefactos/'+image 
        cv2.imwrite(dire, imSinBBOX)
#        print('cls', cls)
#        if cls!=0 and cls!=1 and cls!=2 and cls!=3 and cls!=4 and cls!=5 and cls!=6:
#             plt.imshow(im)
#             plt.show() 
#            re=re+1
    if re > 0 and dm==0:
        print('RE')
        inta=V[y3:y3+h3,x3:x3+w3]
        a,b,ch=inta.shape
        #print(a,b)
#        plt.imshow(V,'Greys')
#        plt.show() 
         
        vecesA = int(a/tamañoA)
        vecesB = int(b/tamañoB)
        
        for f in range(0,a-tamañoA,tamañoA):
            for c in range(0,b-tamañoB,tamañoB):
                #print(f,c)
                cropped2 = inta[f:f+tamañoA,c:c+tamañoB]
               
                #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
                if c==tamañoB*vecesB-tamañoB:
                    cropped2 = inta[f:f+tamañoA,c:]
               
                    #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
                if f==tamañoA*vecesA-tamañoA:
                     #print('ola')
                     if c==tamañoB*vecesB-tamañoB:
                        cropped2 = inta[f:,c:]
               
                         #test2[f:,c:]=test[f:,c:]
                     else:
                         cropped2 = inta[f:,c:c+tamañoB]
                         #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                         #print('dani')
                dire='./RE_ven200x200/'+str(c)+str(f)+'-' +image 
                cv2.imwrite(dire,cropped2)
    if re==0 and dm==0 and imunda==0:
        print('NO')
        inta2=V[y3:y3+h3,x3:x3+w3]
        a,b,ch=inta2.shape
        #print(a,b)
    #        plt.imshow(V,'Greys')
    #        plt.show() 
         
        vecesA = int(a/tamañoA)
        vecesB = int(b/tamañoB)
        
        for f in range(0,a-tamañoA,tamañoA):
            for c in range(0,b-tamañoB,tamañoB):
                #print(f,c)
                cropped3 = inta2[f:f+tamañoA,c:c+tamañoB]
               
                #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
                if c==tamañoB*vecesB-tamañoB:
                    cropped3 = inta2[f:f+tamañoA,c:]
               
                    #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
                if f==tamañoA*vecesA-tamañoA:
                     #print('ola')
                     if c==tamañoB*vecesB-tamañoB:
                        cropped3 = inta2[f:,c:]
               
                         #test2[f:,c:]=test[f:,c:]
                     else:
                         cropped3 = inta2[f:,c:c+tamañoB]
                         #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                         #print('dani')
                dire='./No_ven200x200/'+str(c)+str(f)+'-' +image 
                cv2.imwrite(dire,cropped3)
           
       