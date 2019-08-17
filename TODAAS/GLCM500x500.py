# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:08:48 2019

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
        
energiaDM_LH=[]
homogeneidadDM_LH=[]
correlacionDM_LH=[]
disimiDM_LH=[]
ASMDM_LH=[]
entropiaDM_LH=[]
contrasteDM_LH=[]
mediaglcmDM_LH=[]   
entropianoglcmDM_LH=[]
mediaDM_LH=[]
modaDM_LH=[]    
desviacionDM_LH=[]
curtosisDM_LH=[]
nergianoglcmDM_LH=[]
 ##RE
energiaRE_LH=[]
homogeneidadRE_LH=[]
correlacionRE_LH=[]
disimiRE_LH=[]
ASMRE_LH=[]
entropiaRE_LH=[]
contrasteRE_LH=[]
mediaglcmRE_LH=[]
entropianoglcmRE_LH=[]
mediaRE_LH=[]
modaRE_LH=[]
desviacionRE_LH=[]
curtosisRE_LH=[]
nergianoglcmRE_LH=[]
 ##NO
energiaNO_LH=[]
homogeneidadNO_LH=[]
correlacionNO_LH=[]
disimiNO_LH=[]
ASMNO_LH=[]
entropiaNO_LH=[]
contrasteNO_LH=[]
mediaglcmNO_LH=[]
entropianoglcmNO_LH=[]
mediaNO_LH=[]
modaNO_LH=[]
desviacionNO_LH=[]
curtosisNO_LH=[]
nergianoglcmNO_LH=[]
 
 ##GLCMsinfourierLL
energiaDM_SF_LH=[]
homogeneidadDM_SF_LH=[]
correlacionDM_SF_LH=[]
disimiDM_SF_LH=[]
ASMDM_SF_LH=[]
entropiaDM_SF_LH=[]
contrasteDM_SF_LH=[]
mediaglcmDM_SF_LH=[]
entropianoglcmDM_SF_LH=[]
mediaDM_SF_LH=[]
modaDM_SF_LH=[]
desviacionDM_SF_LH=[]
curtosisDM_SF_LH=[]
nergianoglcmDM_SF_LH=[]
  #RE
energiaRE_SF_LH=[]
homogeneidadRE_SF_LH=[]
correlacionRE_SF_LH=[]
disimiRE_SF_LH=[]
ASMRE_SF_LH=[]
entropiaRE_SF_LH=[]
contrasteRE_SF_LH=[]
mediaglcmRE_SF_LH=[]
entropianoglcmRE_SF_LH=[]
mediaRE_SF_LH=[]
modaRE_SF_LH=[]
desviacionRE_SF_LH=[]
curtosisRE_SF_LH=[]
nergianoglcmRE_SF_LH=[]
 #NO
energiaNO_SF_LH=[]
homogeneidadNO_SF_LH=[]
correlacionNO_SF_LH=[]
disimiNO_SF_LH=[]
ASMNO_SF_LH=[]
entropiaNO_SF_LH=[]
contrasteNO_SF_LH=[]
mediaglcmNO_SF_LH=[]
entropianoglcmNO_SF_LH=[]
mediaNO_SF_LH=[]
modaNO_SF_LH=[]
desviacionNO_SF_LH=[]
curtosisNO_SF_LH=[]
nergianoglcmNO_SF_LH=[]


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
   
        
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            print('DM')
            
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
                       
                             #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
                             #print('dani')
                    #cropFou=cropped
                    cropped_1=cropped.copy()
                    croppedrgb_1=croppedrgb.copy()
                    cropFou=Fourier(cropped)
                    ch= cropFou.shape
                    if len(ch)>2:
                        cropFou=cropFou[:,:,0]
                    
                    g,contraste,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(cropFou)
                    contrasteDM_LH.append(contraste)
                    energiaDM_LH.append(energia)
                    homogeneidadDM_LH.append(homogeneidad)
                    correlacionDM_LH.append(correlacion)
                    disimiDM_LH.append(disimi)
                    ASMDM_LH.append(ASM)
                    entropiaDM_LH.append(entropia)
                    mediaglcmDM_LH.append(np.mean(g))   
                    entropianoglcmDM_LH.append(skimage.measure.shannon_entropy(cropFou))
                    mediaDM_LH.append(np.mean(cropFou))
                    modaDM_LH.append(np.median(g))    
                    desviacionDM_LH.append(np.var(cropFou))
                    curtosisDM_LH.append(sum(kurtosis(cropFou)))
                    nergianoglcmDM_LH.append(np.median(cropFou))
                    """ #Sin Fourier """
                    cropFou_1=cropped_1
                    ch= cropFou_1.shape
                    if len(ch)>2:
                        cropFou_1=cropFou_1[:,:,0]
                    
                    g_1,contraste_1,energia_1,homogeneidad_1, correlacion_1, disimi_1, ASM_1,entropia_1=GLCM(cropFou_1)
                    contrasteDM_SF_LH.append(contraste_1)
                    energiaDM_SF_LH.append(energia_1)
                    homogeneidadDM_SF_LH.append(homogeneidad_1)
                    correlacionDM_SF_LH.append(correlacion_1)
                    disimiDM_SF_LH.append(disimi_1)
                    ASMDM_SF_LH.append(ASM_1)
                    entropiaDM_SF_LH.append(entropia_1)
                    mediaglcmDM_SF_LH.append(np.mean(g_1))   
                    entropianoglcmDM_SF_LH.append(skimage.measure.shannon_entropy(cropFou_1))
                    mediaDM_SF_LH.append(np.mean(cropFou_1))
                    modaDM_SF_LH.append(np.median(g_1))    
                    desviacionDM_SF_LH.append(np.var(cropFou_1))
                    curtosisDM_SF_LH.append(sum(kurtosis(cropFou_1)))
                    nergianoglcmDM_SF_LH.append(np.median(cropFou_1))
                   

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

                    """ #Con Fourier"""
                    cropFou2=Fourier(cropped2)
                    ch= cropFou2.shape
                    if len(ch)>2:
                        cropFou2=cropFou2[:,:,0]
                    
                    g2,contraste2,energia2,homogeneidad2, correlacion2, disimi2, ASM2,entropia2=GLCM(cropFou2)
                    contrasteRE_LH.append(contraste2)
                    energiaRE_LH.append(energia2)
                    homogeneidadRE_LH.append(homogeneidad2)
                    correlacionRE_LH.append(correlacion2)
                    disimiRE_LH.append(disimi2)
                    ASMRE_LH.append(ASM2)
                    entropiaRE_LH.append(entropia2)
                    mediaglcmRE_LH.append(np.mean(g2))   
                    entropianoglcmRE_LH.append(skimage.measure.shannon_entropy(cropFou2))
                    mediaRE_LH.append(np.mean(cropFou2))
                    modaRE_LH.append(np.median(g2))    
                    desviacionRE_LH.append(np.var(cropFou2))
                    curtosisRE_LH.append(sum(kurtosis(cropFou2)))
                    nergianoglcmRE_LH.append(np.median(cropFou2))
                    """ #Sin Fourier"""
                    cropFou2_1= cropped2_1
                    ch= cropFou2_1.shape
                    if len(ch)>2:
                        cropFou2_1=cropFou2_1[:,:,0]
                    
                    g2_1,contraste2_1,energia2_1,homogeneidad2_1, correlacion2_1, disimi2_1, ASM2_1,entropia2_1=GLCM(cropFou2_1)
                    contrasteRE_SF_LH.append(contraste2_1)
                    energiaRE_SF_LH.append(energia2_1)
                    homogeneidadRE_SF_LH.append(homogeneidad2_1)
                    correlacionRE_SF_LH.append(correlacion2_1)
                    disimiRE_SF_LH.append(disimi2_1)
                    ASMRE_SF_LH.append(ASM2_1)
                    entropiaRE_SF_LH.append(entropia2_1)
                    mediaglcmRE_SF_LH.append(np.mean(g2_1))   
                    entropianoglcmRE_SF_LH.append(skimage.measure.shannon_entropy(cropFou2_1))
                    mediaRE_SF_LH.append(np.mean(cropFou2_1))
                    modaRE_SF_LH.append(np.median(g2_1))    
                    desviacionRE_SF_LH.append(np.var(cropFou2_1))
                    curtosisRE_SF_LH.append(sum(kurtosis(cropFou2_1)))
                    nergianoglcmRE_SF_LH.append(np.median(cropFou2_1))
            
    if re==0 and dm==0 and imunda==0:
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

                    """ #Con Fourier"""
                    cropFou3=Fourier(cropped3)
                    ch= cropFou3.shape
                    if len(ch)>2:
                        cropFou3=cropFou3[:,:,0]
                    
                    g3,contraste3,energia3,homogeneidad3, correlacion3, disimi3, ASM3,entropia3=GLCM(cropFou3)
                    contrasteNO_LH.append(contraste3)
                    energiaNO_LH.append(energia3)
                    homogeneidadNO_LH.append(homogeneidad3)
                    correlacionNO_LH.append(correlacion3)
                    disimiNO_LH.append(disimi3)
                    ASMNO_LH.append(ASM3)
                    entropiaNO_LH.append(entropia3)
                    mediaglcmNO_LH.append(np.mean(g3))   
                    entropianoglcmNO_LH.append(skimage.measure.shannon_entropy(cropFou3))
                    mediaNO_LH.append(np.mean(cropFou3))
                    modaNO_LH.append(np.median(g3))    
                    desviacionNO_LH.append(np.var(cropFou3))
                    curtosisNO_LH.append(sum(kurtosis(cropFou3)))
                    nergianoglcmNO_LH.append(np.median(cropFou3))
                    """ #Sin Fourier"""
                    cropFou3_1= cropped3_1
                    ch= cropFou3_1.shape
                    if len(ch)>2:
                        cropFou3_1=cropFou3_1[:,:,0]
                    
                    g3_1,contraste3_1,energia3_1,homogeneidad3_1, correlacion3_1, disimi3_1, ASM3_1,entropia3_1=GLCM(cropFou3_1)
                    contrasteNO_SF_LH.append(contraste3_1)
                    energiaNO_SF_LH.append(energia3_1)
                    homogeneidadNO_SF_LH.append(homogeneidad3_1)
                    correlacionNO_SF_LH.append(correlacion3_1)
                    disimiNO_SF_LH.append(disimi3_1)
                    ASMNO_SF_LH.append(ASM3_1)
                    entropiaNO_SF_LH.append(entropia3_1)
                    mediaglcmNO_SF_LH.append(np.mean(g3_1))   
                    entropianoglcmNO_SF_LH.append(skimage.measure.shannon_entropy(cropFou3_1))
                    mediaNO_SF_LH.append(np.mean(cropFou3_1))
                    modaNO_SF_LH.append(np.median(g3_1))    
                    desviacionNO_SF_LH.append(np.var(cropFou3_1))
                    curtosisNO_SF_LH.append(sum(kurtosis(cropFou3_1)))
                    nergianoglcmNO_SF_LH.append(np.median(cropFou3_1))
                    ###ROJO
                   
         
           
import pandas as pd    
datos = {'EnergiaDM':energiaDM_LH,
         'HomogeneidadDM':homogeneidadDM_LH,
         'CorrelaciónDM':correlacionDM_LH,
         'DisimilitudDM':disimiDM_LH,
         'ASM_DM':ASMDM_LH,
         'EntropíaDM':entropiaDM_LH,
         'ContrasteDM':contrasteDM_LH,
         'Media(glcm)DM':mediaglcmDM_LH,
         'Entropia(no glcm)DM':entropianoglcmDM_LH,
         'MediaDM':mediaDM_LH,
         'MedianaGLCM':modaDM_LH,
         'VarianzaDM':desviacionDM_LH,
         'CurtosisDM':curtosisDM_LH,
         'Energia(no glcm)DM_LH':nergianoglcmDM_LH,
         'Mediana':energiaDM_SF_LH,
         'HomogeneidadDM(sinF)_LH':homogeneidadDM_SF_LH,
         'CorrelaciónDM(sinF)_LH':correlacionDM_SF_LH,
         'DisimilitudDM(sinF)_LH':disimiDM_SF_LH,
         'ASMDM(sinF)_LH':ASMDM_SF_LH,
         'EntropíaDM(sinF)_LH':entropiaDM_SF_LH,
         'ContrasteDM(sinF)_LH':contrasteDM_SF_LH,
         'Media(glcm)DM(sinF)_LH':mediaglcmDM_SF_LH,
         'Entropia(no glcm)DM(sinF)_LH':entropianoglcmDM_SF_LH,
         'MediaDM(sinF)_LH':mediaDM_SF_LH,
         'MedianaGLCM_SF':modaDM_SF_LH,
         'VarianzaDM(sinF)_LH':desviacionDM_SF_LH,
         'CurtosisDM(sinF)_LH':curtosisDM_SF_LH,
         'MedianaSF':nergianoglcmDM_SF_LH}

datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasGLCM_DM.xlsx')             

datos = {'EnergiaRE_LH':energiaRE_LH,
         'HomogeneidadRE_LH':homogeneidadRE_LH,
         'CorrelaciónRE_LH':correlacionRE_LH,
         'DisimilitudRE_LH':disimiRE_LH,
         'ASM_RE_LH':ASMRE_LH,
         'EntropíaRE_LH':entropiaRE_LH,
         'ContrasteRE_LH':contrasteRE_LH,
         'Media(glcm)RE_LH':mediaglcmRE_LH,
         'Entropia(no glcm)RE_LH':entropianoglcmRE_LH,
         'MediaRE_LH':mediaRE_LH,
         'MedianaGLCMRE_LH':modaRE_LH,
         'VarianzaRE_LH':desviacionRE_LH,
         'CurtosisRE_LH':curtosisRE_LH,
         'Mediana(no glcm)RE_LH':nergianoglcmRE_LH,
         'EnergiaRE(sinF)_LH':energiaRE_SF_LH,
         'HomogeneidadRE(sinF)_LH':homogeneidadRE_SF_LH,
         'CorrelaciónRE(sinF)_LH':correlacionRE_SF_LH,
         'DisimilitudRE(sinF)_LH':disimiRE_SF_LH,
         'ASMRE(sinF)_LH':ASMRE_SF_LH,
         'EntropíaRE(sinF)_LH':entropiaRE_SF_LH,
         'ContrasteRE(sinF)_LH':contrasteRE_SF_LH,
         'Media(glcm)RE(sinF)_LH':mediaglcmRE_SF_LH,
         'Entropia(no glcm)RE(sinF)_LH':entropianoglcmRE_SF_LH,
         'MediaRE(sinF)_LH':mediaRE_SF_LH,
         'MedianaGLCMRE(sinF)_LH':modaRE_SF_LH,
         'VarianzaRE(sinF)_LH':desviacionRE_SF_LH,
         'CurtosisRE(sinF)_LH':curtosisRE_SF_LH,
         'mediana(no glcm)RE(sinF)_LH':nergianoglcmRE_SF_LH}
datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasGLCM_RE.xlsx')           
         
         
         ##NO
datos={  'EnergiaNO_LH':energiaNO_LH,
         'HomogeneidadNO_LH':homogeneidadNO_LH,
         'CorrelaciónNO_LH':correlacionNO_LH,
         'DisimilitudNO_LH':disimiNO_LH,
         'ASM_NO_LH':ASMNO_LH,
         'EntropíaNO_LH':entropiaNO_LH,
         'ContrasteNO_LH':contrasteNO_LH,
         'Media(glcm)NO_LH':mediaglcmNO_LH,
         'Entropia(no glcm)NO_LH':entropianoglcmNO_LH,
         'MediaNO_LH':mediaNO_LH,
         'MedianaGLCMNO_LH':modaNO_LH,
         'VarianzanNO_LH':desviacionNO_LH,
         'CurtosisNO_LH':curtosisNO_LH,
         'Mediana(no glcm)NO_LH':nergianoglcmNO_LH,
         'EnergiaNO(sinF)_LH':energiaNO_SF_LH,
         'HomogeneidadNO(sinF)_LH':homogeneidadNO_SF_LH,
         'CorrelaciónNO(sinF)_LH':correlacionNO_SF_LH,
         'DisimilitudNO(sinF)_LH':disimiNO_SF_LH,
         'ASMNO(sinF)_LH':ASMNO_SF_LH,
         'EntropíaNO(sinF)_LH':entropiaNO_SF_LH,
         'ContrasteNO(sinF)_LH':contrasteNO_SF_LH,
         'Media(glcm)NO(sinF)_LH':mediaglcmNO_SF_LH,
         'Entropia(no glcm)NO(sinF)_LH':entropianoglcmNO_SF_LH,
         'MediaNO(sinF)_LH':mediaNO_SF_LH,
         'MedianaGLCMNO(sinF)_LH':modaNO_SF_LH,
         'VarianzaNO(sinF)_LH':desviacionNO_SF_LH,
         'CurtosisNO(sinF)_LH':curtosisNO_SF_LH,
         'Mediana(no glcm)NO(sinF)_LH':nergianoglcmNO_SF_LH}
         
datos = pd.DataFrame(datos)
datos.to_excel('CaracteristicasGLCM_NO.xlsx')               
 