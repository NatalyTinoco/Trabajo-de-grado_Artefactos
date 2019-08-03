# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:51:59 2019

@author: Nataly
"""

import glob
import cv2
import numpy as np 
from rOI import ROI
from normalizacion import log
from filMin import filtrominimo
import pylab as plt 
from matplotlib import pyplot as plt

def eleccionventana(f):
        i=1
        a=1
        divisor=0
        while a!=0:
           if f%i==0 and i>50:
               a=0
               divisor=i
               #print(i)
           else:
               i=i+1
        return divisor
    
for imgfile in glob.glob("*.jpg"):
    ima='./segROI/#5/Z3/'+imgfile
    imaROI=cv2.imread(ima,0)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    img=cv2.imread(imgfile)
    #img=log(img)
    #img=filtrominimo(img)
    f,c,ch=img.shape
    YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y,U,V=cv2.split(YUV)   
    V=Y*imaROI 
    #tavenf=eleccionventana(f)
    #tavenc=eleccionventana(c)
    tavenf=15
    tavenc=15
    #print(tavenf,tavenc)
    #ven=np.zeros((tavenf,tavenc))
    #Ff=0
    #Cc=0
    #ha=tavenf
    #has=tavenc
    
    Binary=V.copy()
    
    image=V
    
    tmp = image # for drawing a rectangle
    stepSize = tavenf
    (w_width, w_height) = (tavenf, tavenc) # window size
    for x in range(0, image.shape[1] - w_width , stepSize):
       for y in range(0, image.shape[0] - w_height, stepSize):
          window = image[x:x + w_width, y:y + w_height]
          #cv2.imshow('W',window)
          #cv2.waitKey(0)
          #cv2.destroyAllWindows()
          #print(x,y)  
          """
          ima=window .reshape(-1)
          plt.ion()
          plt.boxplot([ima], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
          plt.xticks([1], ['img'], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
          plt.ylabel('Img')
          plt.show()
          """
          ta=window.shape
          ta=list(ta)
          binary=window.copy()
          umbral=0.95*np.max(window)
          #print(np.max(ven))
          for f in range(ta[0]):
               for c in range (ta[1]):
                   if window[f,c]<umbral:
                    #if S[f,c]<H[f,c]:
                       binary[f,c]=0
                   else:
                       binary[f,c]=255
          #cv2.imshow('image',binary)
          #cv2.waitKey(0)
          #cv2.destroyAllWindows()
          #print(Binary[x:x + w_width, y:y + w_height].shape)
          Binary[x:x + w_width, y:y + w_height]=binary
     
    fff,ccc=image.shape
    for ff in range(fff):
     for cc in range (ccc):
         if image[ff,cc]<umbral:
          #if S[f,c]<H[f,c]:
             Binary[ff,cc]=0
         else:
             Binary[ff,cc]=255
   
    cv2.imshow('image',Binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
          
    # classify content of the window with your classifier and  
    # determine if the window includes an object (cell) or not
          # draw window on image
          #cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2) # draw rectangle on image
          #plt.imshow(np.array(tmp).astype('uint8'))
    # show all windows
    #plt.show()

        
    """
    for F in range (1,int(f/tavenf)+1):
        for C in range(1,int(c/tavenc)+1):
            ven=V[Ff:has,Cc:ha]
            #V[Ff:has,Cc:ha]=ven[Ff:has,Cc:ha]
            #print(Cc,ha)
            #print(Ff,has)
            #print(F,C)
            #print(V[Ff:ha,Cc:has].shape)
            #plt.imshow(V[Ff:has,Cc:ha],'Greys')
            #plt.show()
            #hist = cv2.calcHist([ven],[0],None,[256],[int(np.min(ven)),int(np.max(ven))])
            #plt.plot(hist)
            #plt.show()
            
            div=8
            nuevohist=hist.tolist() 
            l=int(len(nuevohist)/div)
            i=1
            a=0
            suma=np.zeros((div))
            valm=0
            hasta=0
            menor=0
            for y in range(1,div+1):
                suma[a]=int (sum(np.asarray(nuevohist[i:int(l)*y])))
                if suma[a]>valm:
                    valm=suma[a]
                    hasta=y
                i=int(l)*y+1
                a=a+1
            porcen=0.3
            a=hist[int(l*hasta):]
            a=a.tolist()         
            #print(hasta*l)
            #print(len(a))
            #print(a)
            hist=hist.tolist() 
            u=np.max(a)
            #print(u)
            umbral1 = hist.index(u)
            #print(umbral1)
            
            a=hist[umbral1:]
            #print(a)
            uu=np.min(a)
            histb=hist[umbral1:]          
            umbral=histb.index(uu)
        
            umbral=umbral+len(hist[:umbral1])
            #print(umbral)
            #histc=np.asarray(hist[umbral:])     
            #print(histc)
            #med=np.mean(histc)
            #ii=np.max(histc)
            #print(med)
            #histc=histc.tolist() 
            #uml=histc.index(ii)
            #print(uml)
            #hist = cv2.calcHist([ven],[0],None,[256],[umbral,int(np.max(ven))])
            #plt.plot(hist)
            #plt.show()
            #ima=ven.reshape(-1)
            #plt.ion()
            #plt.boxplot([ima], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
            #plt.xticks([1], ['img'], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
            #plt.ylabel('Img')
            #plt.show()
            #plt.scatter(ven,ven)
            #umbral=umbral+len(hist[:umbral1])+uml
            #print(umbral)

            #cv2.imshow('image',ven)
            #cv2.waitKey(0)
        
            ta=ven.shape
            ta=list(ta)
            binary=ven.copy()
            #print(np.max(ven))
            for x in range(ta[0]):
                for y in range (ta[1]):
                    if ven[x,y]<umbral:
                    #if S[f,c]<H[f,c]:
                        binary[x,y]=0
                    else:
                        binary[x,y]=255
            Binary[Ff:has,Cc:ha]=binary
            #cv2.imshow('image',binary)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
           
            ha=(C+1)*tavenc
            Cc=C*tavenc  
            #print(ven.shape)
        
        has=(F+1)*tavenf
        Ff=F*tavenf
    
    print(imgfile)
    cv2.imshow('image',V)
    cv2.waitKey(0)
    cv2.imshow('image',Binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #"""