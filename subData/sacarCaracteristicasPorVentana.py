# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:18:13 2019

@author: Nataly
"""

#from readimg import read_img
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

for image in glob.glob('*.jpg'):
    # image = '00002.jpg'
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
   
    #cv2.imshow('Grays',imaROI)
    #cv2.destroyAllWindows()
    YUV=cv2.cvtColor(im,cv2.COLOR_RGB2YUV)
    Y,U,V=cv2.split(YUV)
    V=V*imaROI
        
#    for z in range(c):
#        im[:,:,z]=im[:,:,z]*imaROI
#    
    
#    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#    areas = [cv2.contourArea(c) for c in contours]
#    max_index = np.argmax(areas)
#    cnt=contours[max_index]
#    x,y,w,h = cv2.boundingRect(cnt)
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
    imSinBBOX=im.copy()
    tamañoA = 100
    tamañoB = 100
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            print('DM')
            
            dm=dm+1   
            #print(image,dm)
            a,b= V[int(y1):int(y2),int(x1):int(x2)].shape
            #print(a,b)
            #plt.imshow(V[int(y1):int(y2),int(x1):int(x2)],'Greys')
            #plt.show() 
            vecesA = int(a/tamañoA)
            vecesB = int(b/tamañoB)
        
            for f in range(0,a-tamañoA,tamañoA):
                for c in range(0,b-tamañoB,tamañoB):
                    #print(f,c)
                    cropped = V[f:f+tamañoA,c:c+tamañoB]
                   
                    #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
                    if c==tamañoB*vecesB-tamañoB:
                        cropped = V[f:f+tamañoA,c:]
                   
                        #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
                    if f==tamañoA*vecesA-tamañoA:
                         #print('ola')
                         if c==tamañoB*vecesB-tamañoB:
                            cropped = V[f:,c:]
                   
                             #test2[f:,c:]=test[f:,c:]
                         else:
                             cropped = V[f:,c:c+tamañoB]
                             #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                             #print('dani')
                    
                    cropFou=Fourier(cropped)
#                    plt.imshow(cropFou,'Greys')
#                    plt.show()
                    contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(cropFou)
                    contrast.append(contraste)
                    energi.append(energia)
                    homogenei.append(homogeneidad)
                    correlaci.append(correlacion)
                    disi.append(disimi)
                    AS.append(ASM)
                    entrop.append(entropia)
                       
#                    plt.imshow(cropped)
#                    plt.show() 
#                    dire='./cutNotRe_siDM/'+str(c)+str(f)+'-' +image 
#                    cv2.imwrite(dire,cropped)
        if cls==0 and dm==0:
            re=re+1        
            print(re)
        imSinBBOX[int(y1):int(y2),int(x1):int(x2)]=0
        
#        print('cls', cls)
#        if cls!=0 and cls!=1 and cls!=2 and cls!=3 and cls!=4 and cls!=5 and cls!=6:
#             plt.imshow(im)
#             plt.show() 
#            re=re+1
    if re > 0:
#        print('RE')
        a,b=V.shape
        #print(a,b)
#        plt.imshow(V,'Greys')
#        plt.show() 
         
        vecesA = int(a/tamañoA)
        vecesB = int(b/tamañoB)
        
        for f in range(0,a-tamañoA,tamañoA):
            for c in range(0,b-tamañoB,tamañoB):
                #print(f,c)
                cropped = V[f:f+tamañoA,c:c+tamañoB]
               
                #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
                if c==tamañoB*vecesB-tamañoB:
                    cropped = V[f:f+tamañoA,c:]
               
                    #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
                if f==tamañoA*vecesA-tamañoA:
                     #print('ola')
                     if c==tamañoB*vecesB-tamañoB:
                        cropped = V[f:,c:]
               
                         #test2[f:,c:]=test[f:,c:]
                     else:
                         cropped = V[f:,c:c+tamañoB]
                         #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                         #print('dani')
                
                cropFou1=Fourier(cropped)
#                plt.imshow(cropFou1,'Greys')
#                plt.show()
                contraste1,energia1,homogeneidad1, correlacion1, disimi1, ASM1,entropia1=GLCM(cropFou1)
                contrastRE.append(contraste1)
                energiRE.append(energia1)
                homogeneiRE.append(homogeneidad1)
                correlaciRE.append(correlacion1)
                disiRE.append(disimi1)
                ASRE.append(ASM1)
                entropRE.append(entropia1)
       
        
        
table=[contrast,energi,homogenei,correlaci,disi,AS,entrop]       
tableRE=[contrastRE,energiRE,homogeneiRE,correlaciRE,disiRE,ASRE,entropRE]

tabla=['Contraste','Energía','Homogeneidad','Correlación','Disimilitud','ASM','Entropía']

a=0
b=1
h=1

for i in range (len(table)*3):
    print(i)
    print(a,b)
    #with PdfPages('canalV_SinFourier.pdf') as pdf:
    f=plt.figure()
    plt.scatter(table[a][:],table[b][:],c='blue', label='DM')
    plt.scatter(tableRE[a][0:len(table[b][:])],tableRE[b][0:len(table[b][:])],c='red', label='RE')            
    plt.title('GLCM')
    plt.xlabel(tabla[a])
    plt.ylabel(tabla[b])
    plt.legend(loc='top_left')
    plt.show()
    f.savefig('./diagramasDeDis/canalV_ConFourierVentanas/'+str(i))
    #f.savefig("canalV_SinFourier.pdf", bbox_inches='tight')
    plt.close()
    b=b+1
    if b==len(tabla):
        h=h+1
        b=h
        a=a+1
        
tablaDM=[contrast,energi,homogenei,correlaci,disi,AS,entrop]       
tablaRE=[contrastRE,energiRE,homogeneiRE,correlaciRE,disiRE,ASRE,entropRE]

tabla=['Contraste','Energía','Homogeneidad','Correlación','Disimilitud','ASM','Entropía']

a=0
b=1
h=1

for i in range (len(table)):
    f=plt.figure()
    plt.ion()
    plt.boxplot([tablaDM[i][:],tablaRE[i][:]], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
    plt.xticks([1,2], ['DM','RE'], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
    plt.ylabel(tabla[i])
    plt.show()
    f.savefig('./diagramasDeDis/canalV_SinFourier/'+'boxplot'+str(i))
    plt.close()
