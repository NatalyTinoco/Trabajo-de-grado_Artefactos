# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:57:37 2019

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
 
 ##hl
energiaDM_HL=[]
homogeneidadDM_HL=[]
correlacionDM_HL=[]
disimiDM_HL=[]
ASMDM_HL=[]
entropiaDM_HL=[]
contrasteDM_HL=[]
mediaglcmDM_HL=[]
entropianoglcmDM_HL=[]
mediaDM_HL=[]
modaDM_HL=[]
desviacionDM_HL=[]
curtosisDM_HL=[]
nergianoglcmDM_HL=[]
 #RE
energiaRE_HL=[]
homogeneidadRE_HL=[]
correlacionRE_HL=[]
disimiRE_HL=[]
ASMRE_HL=[]
entropiaRE_HL=[]
contrasteRE_HL=[]
mediaglcmRE_HL=[]
entropianoglcmRE_HL=[]
mediaRE_HL=[]
modaRE_HL=[]
desviacionRE_HL=[]
curtosisRE_HL=[]
nergianoglcmRE_HL=[]
 #NO
energiaNO_HL=[]
homogeneidadNO_HL=[]
correlacionNO_HL=[]
disimiNO_HL=[]
ASMNO_HL=[]
entropiaNO_HL=[]
contrasteNO_HL=[]
mediaglcmNO_HL=[]
entropianoglcmNO_HL=[]
mediaNO_HL=[]
modaNO_HL=[]
desviacionNO_HL=[]
curtosisNO_HL=[]
nergianoglcmNO_HL=[]
 
 
 ##GLCMsinfourierLL
energiaDM_SF_HL=[]
homogeneidadDM_SF_HL=[]
correlacionDM_SF_HL=[]
disimiDM_SF_HL=[]
ASMDM_SF_HL=[]
entropiaDM_SF_HL=[]
contrasteDM_SF_HL=[]
mediaglcmDM_SF_HL=[]
entropianoglcmDM_SF_HL=[]
mediaDM_SF_HL=[]
modaDM_SF_HL=[]
desviacionDM_SF_HL=[]
curtosisDM_SF_HL=[]
nergianoglcmDM_SF_HL=[]
 #RE
energiaRE_SF_HL=[]
homogeneidadRE_SF_HL=[]
correlacionRE_SF_HL=[]
disimiRE_SF_HL=[]
ASMRE_SF_HL=[]
entropiaRE_SF_HL=[]
contrasteRE_SF_HL=[]
mediaglcmRE_SF_HL=[]
entropianoglcmRE_SF_HL=[]
mediaRE_SF_HL=[]
modaRE_SF_HL=[]
desviacionRE_SF_HL=[]
curtosisRE_SF_HL=[]
nergianoglcmRE_SF_HL=[]
#NO
energiaNO_SF_HL=[]
homogeneidadNO_SF_HL=[]
correlacionNO_SF_HL=[]
disimiNO_SF_HL=[]
ASMNO_SF_HL=[]
entropiaNO_SF_HL=[]
contrasteNO_SF_HL=[]
mediaglcmNO_SF_HL=[]
entropianoglcmNO_SF_HL=[]
mediaNO_SF_HL=[]
modaNO_SF_HL=[]
desviacionNO_SF_HL=[]
curtosisNO_SF_HL=[]
nergianoglcmNO_SF_HL=[]
 
 
 
 ## HLrojoo
energiaDM_HLROJO=[]
homogeneidadDM_HLROJO=[]
correlacionDM_HLROJO=[]
disimiDM_HLROJO=[]
ASMDM_HLROJO=[]
entropiaDM_HLROJO=[]
contrasteDM_HLROJO=[]
mediaglcmDM_HLROJO=[]
entropianoglcmDM_HLROJO=[]
mediaDM_HLROJO=[]
modaDM_HLROJO=[]
desviacionDM_HLROJO=[]
curtosisDM_HLROJO=[]
nergianoglcmDM_HLROJO=[]
 #RE
energiaRE_HLROJO=[]
homogeneidadRE_HLROJO=[]
correlacionRE_HLROJO=[]
disimiRE_HLROJO=[]
ASMRE_HLROJO=[]
entropiaRE_HLROJO=[]
contrasteRE_HLROJO=[]
mediaglcmRE_HLROJO=[]
entropianoglcmRE_HLROJO=[]
mediaRE_HLROJO=[]
modaRE_HLROJO=[]
desviacionRE_HLROJO=[]
curtosisRE_HLROJO=[]
nergianoglcmRE_HLROJO=[]
 #NO
energiaNO_HLROJO=[]
homogeneidadNO_HLROJO=[]
correlacionNO_HLROJO=[]
disimiNO_HLROJO=[]
ASMNO_HLROJO=[]
entropiaNO_HLROJO=[]
contrasteNO_HLROJO=[]
mediaglcmNO_HLROJO=[]
entropianoglcmNO_HLROJO=[]
mediaNO_HLROJO=[]
modaNO_HLROJO=[]
desviacionNO_HLROJO=[]
curtosisNO_HLROJO=[]
nergianoglcmNO_HLROJO=[]
 
 
##GLCMsinfourierLL
energiaDM_SF_HLROJO=[]
homogeneidadDM_SF_HLROJO=[]
correlacionDM_SF_HLROJO=[]
disimiDM_SF_HLROJO=[]
ASMDM_SF_HLROJO=[]
entropiaDM_SF_HLROJO=[]
contrasteDM_SF_HLROJO=[]
mediaglcmDM_SF_HLROJO=[]
entropianoglcmDM_SF_HLROJO=[]
mediaDM_SF_HLROJO=[]
modaDM_SF_HLROJO=[]
desviacionDM_SF_HLROJO=[]
curtosisDM_SF_HLROJO=[]
nergianoglcmDM_SF_HLROJO=[]         
 #RE
energiaRE_SF_HLROJO=[]
homogeneidadRE_SF_HLROJO=[]
correlacionRE_SF_HLROJO=[]
disimiRE_SF_HLROJO=[]
ASMRE_SF_HLROJO=[]
entropiaRE_SF_HLROJO=[]
contrasteRE_SF_HLROJO=[]
mediaglcmRE_SF_HLROJO=[]
entropianoglcmRE_SF_HLROJO=[]
mediaRE_SF_HLROJO=[]
modaRE_SF_HLROJO=[]
desviacionRE_SF_HLROJO=[]
curtosisRE_SF_HLROJO=[]
nergianoglcmRE_SF_HLROJO=[]         
#NO
energiaNO_SF_HLROJO=[]
homogeneidadNO_SF_HLROJO=[]
correlacionNO_SF_HLROJO=[]
disimiNO_SF_HLROJO=[]
ASMNO_SF_HLROJO=[]
entropiaNO_SF_HLROJO=[]
contrasteNO_SF_HLROJO=[]
mediaglcmNO_SF_HLROJO=[]
entropianoglcmNO_SF_HLROJO=[]
mediaNO_SF_HLROJO=[]
modaNO_SF_HLROJO=[]
desviacionNO_SF_HLROJO=[]
curtosisNO_SF_HLROJO=[]
nergianoglcmNO_SF_HLROJO=[]         
  


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

                    """ #Con Fourier No rojo"""
                    coeffs2 = pywt.dwt2(cropped, 'bior1.3')
                    LL, (LH, HL, HH) = coeffs2
                    LH=LH.astype(np.uint8)
                    cropFou=Fourier(LH)
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
                    #modaDM_LH.append(np.mean(stats.mode(cropFou)))    
                    desviacionDM_LH.append(np.var(cropFou))
                    curtosisDM_LH.append(kurtosis(cropFou))
                    nergianoglcmDM_LH.append(np.median(cropFou))
                    """ #Sin Fourier No rojo"""
                    coeffs2_1 = pywt.dwt2(cropped_1, 'bior1.3')
                    LL_1, (LH_1, HL_1, HH_1) = coeffs2_1
                    LH_1=LH_1.astype(np.uint8)
                    cropFou_1=LH_1
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
                    #modaDM_SF_LH.append(np.mean(stats.mode(cropFou_1)))    
                    desviacionDM_SF_LH.append(np.var(cropFou_1))
                    curtosisDM_SF_LH.append(kurtosis(cropFou_1))
                    nergianoglcmDM_SF_LH.append(np.median(cropFou_1))
                    ###ROJO
                    """ #Con Fourier  rojo"""
                    coeffs2rgb = pywt.dwt2(croppedrgb, 'bior1.3')
                    LLrgb, (LHrgb, HLrgb, HHrgb) = coeffs2rgb
                    HLrgb=HLrgb.astype(np.uint8)
                    cropFourgb=Fourier(HLrgb)
                    ch= cropFourgb.shape
                    if len(ch)>2:
                        cropFourgb=cropFourgb[:,:,0]
                    
                    grgb,contrastergb,energiargb,homogeneidadrgb, correlacionrgb, disimirgb, ASMrgb,entropiargb=GLCM(cropFourgb)
                    contrasteDM_HLROJO.append(contrastergb)
                    energiaDM_HLROJO.append(energiargb)
                    homogeneidadDM_HLROJO.append(homogeneidadrgb)
                    correlacionDM_HLROJO.append(correlacionrgb)
                    disimiDM_HLROJO.append(disimirgb)
                    ASMDM_HLROJO.append(ASMrgb)
                    entropiaDM_HLROJO.append(entropiargb)
                    mediaglcmDM_HLROJO.append(np.mean(grgb))   
                    entropianoglcmDM_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb))
                    mediaDM_HLROJO.append(np.mean(cropFourgb))
                    #modaDM_HLROJO.append(np.mean(stats.mode(cropFourgb)))    
                    desviacionDM_HLROJO.append(np.var(cropFourgb))
                    curtosisDM_HLROJO.append(kurtosis(cropFourgb))
                    nergianoglcmDM_HLROJO.append(np.median(cropFourgb))
                    """ #Sin Fourier rojo"""
                    coeffs2rgb_1 = pywt.dwt2(croppedrgb_1, 'bior1.3')
                    LLrgb_1, (LHrgb_1, HLrgb_1, HHrgb_1) = coeffs2rgb_1
                    HLrgb_1=HLrgb_1.astype(np.uint8)
                    cropFourgb_1=HLrgb_1
                    ch=cropFourgb_1.shape
                    if len(ch)>2:
                        cropFourgb_1=cropFourgb_1[:,:,0]
                    
                    grgb_1,contrastergb_1,energiargb_1,homogeneidadrgb_1, correlacionrgb_1, disimirgb_1, ASMrgb_1,entropiargb_1=GLCM(cropFourgb_1)
                    contrasteDM_SF_HLROJO.append(contrastergb_1)
                    energiaDM_SF_HLROJO.append(energiargb_1)
                    homogeneidadDM_SF_HLROJO.append(homogeneidadrgb_1)
                    correlacionDM_SF_HLROJO.append(correlacionrgb_1)
                    disimiDM_SF_HLROJO.append(disimirgb_1)
                    ASMDM_SF_HLROJO.append(ASMrgb_1)
                    entropiaDM_SF_HLROJO.append(entropiargb_1)
                    mediaglcmDM_SF_HLROJO.append(np.mean(grgb_1))   
                    entropianoglcmDM_SF_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb_1))
                    mediaDM_SF_HLROJO.append(np.mean(cropFourgb_1))
                    #modaDM_SF_HLROJO.append(np.mean(stats.mode(cropFourgb_1)))    
                    desviacionDM_SF_HLROJO.append(np.var(cropFourgb_1))
                    curtosisDM_SF_HLROJO.append(kurtosis(cropFourgb_1))
                    nergianoglcmDM_SF_HLROJO.append(np.median(cropFourgb_1))

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

                    """ #Con Fourier No rojo"""
                    coeffs22 = pywt.dwt2(cropped2, 'bior1.3')
                    LL2, (LH2, HL2, HH2) = coeffs22
                    LH2=LH2.astype(np.uint8)
                    cropFou2=Fourier(LH2)
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
                    #modaRE_LH.append(np.mean(stats.mode(list(cropFou2))))    
                    desviacionRE_LH.append(np.var(cropFou2))
                    curtosisRE_LH.append(kurtosis(cropFou2))
                    nergianoglcmRE_LH.append(np.median(cropFou2))
                    """ #Sin Fourier No rojo"""
                    coeffs22_1 = pywt.dwt2(cropped2_1, 'bior1.3')
                    LL2_1, (LH2_1, HL2_1, HH2_1) = coeffs22_1
                    LH2_1=LH2_1.astype(np.uint8)
                    cropFou2_1=LH2_1
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
                    #modaRE_SF_LH.append(np.mean(stats.mode(cropFou2_1)))    
                    desviacionRE_SF_LH.append(np.var(cropFou2_1))
                    curtosisRE_SF_LH.append(kurtosis(cropFou2_1))
                    nergianoglcmRE_SF_LH.append(np.median(cropFou2_1))
                    ###ROJO
                    """ #Con Fourier  rojo"""
                    coeffs22rgb = pywt.dwt2(croppedrgb2, 'bior1.3')
                    LLrgb2, (LHrgb2, HLrgb2, HHrgb2) = coeffs22rgb
                    HLrgb2=HLrgb2.astype(np.uint8)
                    cropFourgb2=Fourier(HLrgb2)
                    ch= cropFourgb2.shape
                    if len(ch)>2:
                        cropFourgb2=cropFourgb2[:,:,0]
                    
                    grgb2,contrastergb2,energiargb2,homogeneidadrgb2, correlacionrgb2, disimirgb2, ASMrgb2,entropiargb2=GLCM(cropFourgb2)
#                    plt.imshow(cropFourgb2[:,:,0])
#                    plt.show()
                    contrasteRE_HLROJO.append(contrastergb2)
                    energiaRE_HLROJO.append(energiargb2)
                    homogeneidadRE_HLROJO.append(homogeneidadrgb2)
                    correlacionRE_HLROJO.append(correlacionrgb2)
                    disimiRE_HLROJO.append(disimirgb2)
                    ASMRE_HLROJO.append(ASMrgb2)
                    entropiaRE_HLROJO.append(entropiargb2)
                    mediaglcmRE_HLROJO.append(np.mean(grgb2))   
                    entropianoglcmRE_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb2))
                    mediaRE_HLROJO.append(np.mean(cropFourgb2))
                    #modaRE_HLROJO.append(np.mean(stats.mode(cropFourgb2)))    
                    desviacionRE_HLROJO.append(np.var(cropFourgb2))
                    curtosisRE_HLROJO.append(kurtosis(cropFourgb2))
                    nergianoglcmRE_HLROJO.append(np.median(cropFourgb2))
                    """ #Sin Fourier rojo"""
                    coeffs2rgb2_1 = pywt.dwt2(croppedrgb2_1, 'bior1.3')
                    LLrgb2_1, (LHrgb2_1, HLrgb2_1, HHrgb2_1) = coeffs2rgb2_1
                    HLrgb2_1=HLrgb2_1.astype(np.uint8)
                    cropFourgb2_1=HLrgb2_1
                    ch= cropFourgb2_1.shape
                    if len(ch)>2:
                        cropFourgb2_1=cropFourgb2_1[:,:,0]
                    
                    grgb2_1,contrastergb2_1,energiargb2_1,homogeneidadrgb2_1, correlacionrgb2_1, disimirgb2_1, ASMrgb2_1,entropiargb2_1=GLCM(cropFourgb2_1)
                    contrasteRE_SF_HLROJO.append(contrastergb2_1)
                    energiaRE_SF_HLROJO.append(energiargb2_1)
                    homogeneidadRE_SF_HLROJO.append(homogeneidadrgb2_1)
                    correlacionRE_SF_HLROJO.append(correlacionrgb2_1)
                    disimiRE_SF_HLROJO.append(disimirgb2_1)
                    ASMRE_SF_HLROJO.append(ASMrgb2_1)
                    entropiaRE_SF_HLROJO.append(entropiargb2_1)
                    mediaglcmRE_SF_HLROJO.append(np.mean(grgb2_1))   
                    entropianoglcmRE_SF_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb2_1))
                    mediaRE_SF_HLROJO.append(np.mean(cropFourgb2_1))
                    #modaRE_SF_HLROJO.append(np.mean(stats.mode(cropFourgb2_1)))    
                    desviacionRE_SF_HLROJO.append(np.var(cropFourgb2_1))
                    curtosisRE_SF_HLROJO.append(kurtosis(cropFourgb2_1))
                    nergianoglcmRE_SF_HLROJO.append(np.median(cropFourgb2_1))

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

                    """ #Con Fourier No rojo"""
                    coeffs23 = pywt.dwt2(cropped3, 'bior1.3')
                    LL3, (LH3, HL3, HH3) = coeffs23
                    LH3=LH3.astype(np.uint8)
                    cropFou3=Fourier(LH3)
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
                    #modaNO_LH.append(np.mean(stats.mode(cropFou3)))    
                    desviacionNO_LH.append(np.var(cropFou3))
                    curtosisNO_LH.append(kurtosis(cropFou3))
                    nergianoglcmNO_LH.append(np.median(cropFou3))
                    """ #Sin Fourier No rojo"""
                    coeffs23_1 = pywt.dwt2(cropped3_1, 'bior1.3')
                    LL2_1, (LH3_1, HL2_1, HH2_1) = coeffs23_1
                    LH3_1=LH3_1.astype(np.uint8)
                    cropFou3_1=LH3_1
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
                    #modaNO_SF_LH.append(np.mean(stats.mode(cropFou3_1)))    
                    desviacionNO_SF_LH.append(np.var(cropFou3_1))
                    curtosisNO_SF_LH.append(kurtosis(cropFou3_1))
                    nergianoglcmNO_SF_LH.append(np.median(cropFou3_1))
                    ###ROJO
                    """ #Con Fourier  rojo"""
                    coeffs23rgb = pywt.dwt2(croppedrgb3, 'bior1.3')
                    LLrgb2, (LHrgb2, HLrgb3, HHrgb2) = coeffs23rgb
                    HLrgb3=HLrgb3.astype(np.uint8)
                    cropFourgb3=Fourier(HLrgb3)
                    ch= cropFourgb3.shape
                    if len(ch)>2:
                        cropFourgb3=cropFourgb3[:,:,0]
                    
                    grgb3,contrastergb3,energiargb3,homogeneidadrgb3, correlacionrgb3, disimirgb3, ASMrgb3,entropiargb3=GLCM(cropFourgb3)
                    contrasteNO_HLROJO.append(contrastergb3)
                    energiaNO_HLROJO.append(energiargb3)
                    homogeneidadNO_HLROJO.append(homogeneidadrgb3)
                    correlacionNO_HLROJO.append(correlacionrgb3)
                    disimiNO_HLROJO.append(disimirgb3)
                    ASMNO_HLROJO.append(ASMrgb3)
                    entropiaNO_HLROJO.append(entropiargb3)
                    mediaglcmNO_HLROJO.append(np.mean(grgb3))   
                    entropianoglcmNO_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb3))
                    mediaNO_HLROJO.append(np.mean(cropFourgb3))
                    #modaNO_HLROJO.append(np.mean(stats.mode(cropFourgb3)))    
                    desviacionNO_HLROJO.append(np.var(cropFourgb3))
                    curtosisNO_HLROJO.append(kurtosis(cropFourgb2))
                    nergianoglcmNO_HLROJO.append(np.median(cropFourgb3))
                    """ #Sin Fourier rojo"""
                    coeffs2rgb3_1 = pywt.dwt2(croppedrgb3_1, 'bior1.3')
                    LLrgb2_1, (LHrgb2_1, HLrgb3_1, HHrgb2_1) = coeffs2rgb2_1
                    HLrgb3_1=HLrgb3_1.astype(np.uint8)
                    cropFourgb3_1=HLrgb3_1
                    ch= cropFourgb3_1.shape
                    if len(ch)>2:
                        cropFourgb3_1=cropFourgb3_1[:,:,0]
                    
                    grgb3_1,contrastergb3_1,energiargb3_1,homogeneidadrgb3_1, correlacionrgb3_1, disimirgb3_1, ASMrgb3_1,entropiargb3_1=GLCM(cropFourgb3_1)
                    contrasteNO_SF_HLROJO.append(contrastergb3_1)
                    energiaNO_SF_HLROJO.append(energiargb3_1)
                    homogeneidadNO_SF_HLROJO.append(homogeneidadrgb3_1)
                    correlacionNO_SF_HLROJO.append(correlacionrgb3_1)
                    disimiNO_SF_HLROJO.append(disimirgb3_1)
                    ASMNO_SF_HLROJO.append(ASMrgb3_1)
                    entropiaNO_SF_HLROJO.append(entropiargb3_1)
                    mediaglcmNO_SF_HLROJO.append(np.mean(grgb3_1))   
                    entropianoglcmNO_SF_HLROJO.append(skimage.measure.shannon_entropy(cropFourgb3_1))
                    mediaNO_SF_HLROJO.append(np.mean(cropFourgb3_1))
                    #modaNO_SF_HLROJO.append(np.mean(stats.mode(cropFourgb3_1)))    
                    desviacionNO_SF_HLROJO.append(np.var(cropFourgb3_1))
                    curtosisNO_SF_HLROJO.append(kurtosis(cropFourgb3_1))
                    nergianoglcmNO_SF_HLROJO.append(np.median(cropFourgb3_1))
         
           
import pandas as pd    
datos = {'EnergiaDM_LH':energiaDM_LH,
         'HomogeneidadDM_LH':homogeneidadDM_LH,
         'CorrelaciónDM_LH':correlacionDM_LH,
         'DisimilitudDM_LH':disimiDM_LH,
         'ASM_DM_LH':ASMDM_LH,
         'EntropíaDM_LH':entropiaDM_LH,
         'ContrasteDM_LH':contrasteDM_LH,
         'Media(glcm)DM_LH':mediaglcmDM_LH,
         'Entropia(no glcm)DM_LH':entropianoglcmDM_LH,
         'MediaDM_LH':mediaDM_LH,
         'ModaDM_LH':modaDM_LH,
         'DesviacionDM_LH':desviacionDM_LH,
         'CurtosisDM_LH':curtosisDM_LH,
         'Energia(no glcm)DM_LH':nergianoglcmDM_LH,
         ##RE
         'EnergiaRE_LH':energiaRE_LH,
         'HomogeneidadRE_LH':homogeneidadRE_LH,
         'CorrelaciónRE_LH':correlacionRE_LH,
         'DisimilitudRE_LH':disimiRE_LH,
         'ASM_RE_LH':ASMRE_LH,
         'EntropíaRE_LH':entropiaRE_LH,
         'ContrasteRE_LH':contrasteRE_LH,
         'Media(glcm)RE_LH':mediaglcmRE_LH,
         'Entropia(no glcm)RE_LH':entropianoglcmRE_LH,
         'MediaRE_LH':mediaRE_LH,
         'ModaRE_LH':modaRE_LH,
         'DesviacionRE_LH':desviacionRE_LH,
         'CurtosisRE_LH':curtosisRE_LH,
         'Energia(no glcm)RE_LH':nergianoglcmRE_LH,
         ##NO
         'EnergiaNO_LH':energiaNO_LH,
         'HomogeneidadNO_LH':homogeneidadNO_LH,
         'CorrelaciónNO_LH':correlacionNO_LH,
         'DisimilitudNO_LH':disimiNO_LH,
         'ASM_NO_LH':ASMNO_LH,
         'EntropíaNO_LH':entropiaNO_LH,
         'ContrasteNO_LH':contrasteNO_LH,
         'Media(glcm)NO_LH':mediaglcmNO_LH,
         'Entropia(no glcm)NO_LH':entropianoglcmNO_LH,
         'MediaNO_LH':mediaNO_LH,
         'ModaNO_LH':modaNO_LH,
         'DesviacionNO_LH':desviacionNO_LH,
         'CurtosisNO_LH':curtosisNO_LH,
         'Energia(no glcm)NO_LH':nergianoglcmNO_LH,
         
         ##GLCMsinfourierLL
         'EnergiaDM(sinF)_LH':energiaDM_SF_LH,
         'HomogeneidadDM(sinF)_LH':homogeneidadDM_SF_LH,
         'CorrelaciónDM(sinF)_LH':correlacionDM_SF_LH,
         'DisimilitudDM(sinF)_LH':disimiDM_SF_LH,
         'ASMDM(sinF)_LH':ASMDM_SF_LH,
         'EntropíaDM(sinF)_LH':entropiaDM_SF_LH,
         'ContrasteDM(sinF)_LH':contrasteDM_SF_LH,
         'Media(glcm)DM(sinF)_LH':mediaglcmDM_SF_LH,
         'Entropia(no glcm)DM(sinF)_LH':entropianoglcmDM_SF_LH,
         'MediaDM(sinF)_LH':mediaDM_SF_LH,
         'ModaDM(sinF)_LH':modaDM_SF_LH,
         'DesviacionDM(sinF)_LH':desviacionDM_SF_LH,
         'CurtosisDM(sinF)_LH':curtosisDM_SF_LH,
         'Energia(no glcm)DM(sinF)_LH':nergianoglcmDM_SF_LH,
          #RE
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
         'ModaRE(sinF)_LH':modaRE_SF_LH,
         'DesviacionRE(sinF)_LH':desviacionRE_SF_LH,
         'CurtosisRE(sinF)_LH':curtosisRE_SF_LH,
         'Energia(no glcm)RE(sinF)_LH':nergianoglcmRE_SF_LH,
         #NO
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
         'ModaNO(sinF)_LH':modaNO_SF_LH,
         'DesviacionNO(sinF)_LH':desviacionNO_SF_LH,
         'CurtosisNO(sinF)_LH':curtosisNO_SF_LH,
         'Energia(no glcm)NO(sinF)_LH':nergianoglcmNO_SF_LH,
         
         ##hl
         'EnergiaDM_HL':energiaDM_HL,
         'HomogeneidadDM_HL':homogeneidadDM_HL,
         'CorrelaciónDM_HL':correlacionDM_HL,
         'DisimilitudDM_HL':disimiDM_HL,
         'ASM_DM_HL':ASMDM_HL,
         'EntropíaDM_HL':entropiaDM_HL,
         'ContrasteDM_HL':contrasteDM_HL,
         'Media(glcm)DM_HL':mediaglcmDM_HL,
         'Entropia(no glcm)DM_HL':entropianoglcmDM_HL,
         'MediaDM_HL':mediaDM_HL,
         'ModaDM_HL':modaDM_HL,
         'DesviacionDM_HL':desviacionDM_HL,
         'CurtosisDM_HL':curtosisDM_HL,
         'Energia(no glcm)DM_HL':nergianoglcmDM_HL,
         #RE
         'EnergiaRE_HL':energiaRE_HL,
         'HomogeneidadRE_HL':homogeneidadRE_HL,
         'CorrelaciónRE_HL':correlacionRE_HL,
         'DisimilitudRE_HL':disimiRE_HL,
         'ASM_RE_HL':ASMRE_HL,
         'EntropíaRE_HL':entropiaRE_HL,
         'ContrasteRE_HL':contrasteRE_HL,
         'Media(glcm)RE_HL':mediaglcmRE_HL,
         'Entropia(no glcm)RE_HL':entropianoglcmRE_HL,
         'MediaRE_HL':mediaRE_HL,
         'ModaRE_HL':modaRE_HL,
         'DesviacionRE_HL':desviacionRE_HL,
         'CurtosisRE_HL':curtosisRE_HL,
         'Energia(no glcm)RE_HL':nergianoglcmRE_HL,
         #NO
         'EnergiaNO_HL':energiaNO_HL,
         'HomogeneidadNO_HL':homogeneidadNO_HL,
         'CorrelaciónNO_HL':correlacionNO_HL,
         'DisimilitudNO_HL':disimiNO_HL,
         'ASM_NO_HL':ASMNO_HL,
         'EntropíaNO_HL':entropiaNO_HL,
         'ContrasteNO_HL':contrasteNO_HL,
         'Media(glcm)NO_HL':mediaglcmNO_HL,
         'Entropia(no glcm)NO_HL':entropianoglcmNO_HL,
         'MediaNO_HL':mediaNO_HL,
         'ModaNO_HL':modaNO_HL,
         'DesviacionNO_HL':desviacionNO_HL,
         'CurtosisNO_HL':curtosisNO_HL,
         'Energia(no glcm)NO_HL':nergianoglcmNO_HL,
         
         
         ##GLCMsinfourierLL
         'EnergiaDM(sinF)_HL':energiaDM_SF_HL,
         'HomogeneidadDM(sinF)_HL':homogeneidadDM_SF_HL,
         'CorrelaciónDM(sinF)_HL':correlacionDM_SF_HL,
         'DisimilitudDM(sinF)_HL':disimiDM_SF_HL,
         'ASMDM(sinF)_HL':ASMDM_SF_HL,
         'EntropíaDM(sinF)_HL':entropiaDM_SF_HL,
         'ContrasteDM(sinF)_HL':contrasteDM_SF_HL,
         'Media(glcm)DM(sinF)_HL':mediaglcmDM_SF_HL,
         'Entropia(no glcm)DM(sinF)_HL':entropianoglcmDM_SF_HL,
         'MediaDM(sinF)_HL':mediaDM_SF_HL,
         'ModaDM(sinF)_HL':modaDM_SF_HL,
         'DesviacionDM(sinF)_HL':desviacionDM_SF_HL,
         'CurtosisDM(sinF)_HL':curtosisDM_SF_HL,
         'Energia(no glcm)DM(sinF)_HL':nergianoglcmDM_SF_HL,
         #RE
         'EnergiaRE(sinF)_HL':energiaRE_SF_HL,
         'HomogeneidadRE(sinF)_HL':homogeneidadRE_SF_HL,
         'CorrelaciónRE(sinF)_HL':correlacionRE_SF_HL,
         'DisimilitudRE(sinF)_HL':disimiRE_SF_HL,
         'ASMRE(sinF)_HL':ASMRE_SF_HL,
         'EntropíaRE(sinF)_HL':entropiaRE_SF_HL,
         'ContrasteRE(sinF)_HL':contrasteRE_SF_HL,
         'Media(glcm)RE(sinF)_HL':mediaglcmRE_SF_HL,
         'Entropia(no glcm)RE(sinF)_HL':entropianoglcmRE_SF_HL,
         'MediaRE(sinF)_HL':mediaRE_SF_HL,
         'ModaRE(sinF)_HL':modaRE_SF_HL,
         'DesviacionRE(sinF)_HL':desviacionRE_SF_HL,
         'CurtosisRE(sinF)_HL':curtosisRE_SF_HL,
         'Energia(no glcm)RE(sinF)_HL':nergianoglcmRE_SF_HL,
          #NO
         'EnergiaNO(sinF)_HL':energiaNO_SF_HL,
         'HomogeneidadNO(sinF)_HL':homogeneidadNO_SF_HL,
         'CorrelaciónNO(sinF)_HL':correlacionNO_SF_HL,
         'DisimilitudNO(sinF)_HL':disimiNO_SF_HL,
         'ASMNO(sinF)_HL':ASMNO_SF_HL,
         'EntropíaNO(sinF)_HL':entropiaNO_SF_HL,
         'ContrasteNO(sinF)_HL':contrasteNO_SF_HL,
         'Media(glcm)NO(sinF)_HL':mediaglcmNO_SF_HL,
         'Entropia(no glcm)NO(sinF)_HL':entropianoglcmNO_SF_HL,
         'MediaNO(sinF)_HL':mediaNO_SF_HL,
         'ModaNO(sinF)_HL':modaNO_SF_HL,
         'DesviacionNO(sinF)_HL':desviacionNO_SF_HL,
         'CurtosisNO(sinF)_HL':curtosisNO_SF_HL,
         'Energia(no glcm)NO(sinF)_HL':nergianoglcmNO_SF_HL,
         
         
         
         ## HLrojoo
         'EnergiaDM_HLROJO':energiaDM_HLROJO,
         'HomogeneidadDM_HLROJO':homogeneidadDM_HLROJO,
         'CorrelaciónDM_HLROJO':correlacionDM_HLROJO,
         'DisimilitudDM_HLROJO':disimiDM_HLROJO,
         'ASM_DM_HLROJO':ASMDM_HLROJO,
         'EntropíaDM_HLROJO':entropiaDM_HLROJO,
         'ContrasteDM_HLROJO':contrasteDM_HLROJO,
         'Media(glcm)DM_HLROJO':mediaglcmDM_HLROJO,
         'Entropia(no glcm)DM_HLROJO':entropianoglcmDM_HLROJO,
         'MediaDM_HLROJO':mediaDM_HLROJO,
         'ModaDM_HLROJO':modaDM_HLROJO,
         'DesviacionDM_HLROJO':desviacionDM_HLROJO,
         'CurtosisDM_HLROJO':curtosisDM_HLROJO,
         'Energia(no glcm)DM_HLROJO':nergianoglcmDM_HLROJO,
         #RE
         'EnergiaRE_HLROJO':energiaRE_HLROJO,
         'HomogeneidadRE_HLROJO':homogeneidadRE_HLROJO,
         'CorrelaciónRE_HLROJO':correlacionRE_HLROJO,
         'DisimilitudRE_HLROJO':disimiRE_HLROJO,
         'ASM_RE_HLROJO':ASMRE_HLROJO,
         'EntropíaRE_HLROJO':entropiaRE_HLROJO,
         'ContrasteRE_HLROJO':contrasteRE_HLROJO,
         'Media(glcm)RE_HLROJO':mediaglcmRE_HLROJO,
         'Entropia(no glcm)RE_HLROJO':entropianoglcmRE_HLROJO,
         'MediaRE_HLROJO':mediaRE_HLROJO,
         'ModaRE_HLROJO':modaRE_HLROJO,
         'DesviacionRE_HLROJO':desviacionRE_HLROJO,
         'CurtosisRE_HLROJO':curtosisRE_HLROJO,
         'Energia(no glcm)RE_HLROJO':nergianoglcmRE_HLROJO,
         #NO
         'EnergiaNO_HLROJO':energiaNO_HLROJO,
         'HomogeneidadNO_HLROJO':homogeneidadNO_HLROJO,
         'CorrelaciónNO_HLROJO':correlacionNO_HLROJO,
         'DisimilitudNO_HLROJO':disimiNO_HLROJO,
         'ASM_NO_HLROJO':ASMNO_HLROJO,
         'EntropíaNO_HLROJO':entropiaNO_HLROJO,
         'ContrasteNO_HLROJO':contrasteNO_HLROJO,
         'Media(glcm)NO_HLROJO':mediaglcmNO_HLROJO,
         'Entropia(no glcm)NO_HLROJO':entropianoglcmNO_HLROJO,
         'MediaNO_HLROJO':mediaNO_HLROJO,
         'ModaNO_HLROJO':modaNO_HLROJO,
         'DesviacionNO_HLROJO':desviacionNO_HLROJO,
         'CurtosisNO_HLROJO':curtosisNO_HLROJO,
         'Energia(no glcm)NO_HLROJO':nergianoglcmNO_HLROJO,
         
         
         ##GLCMsinfourierLL
         'EnergiaDM(sinF)_HLROJO':energiaDM_SF_HLROJO,
         'HomogeneidadDM(sinF)_HLROJO':homogeneidadDM_SF_HLROJO,
         'CorrelaciónDM(sinF)_HLROJO':correlacionDM_SF_HLROJO,
         'DisimilitudDM(sinF)_HLROJO':disimiDM_SF_HLROJO,
         'ASMDM(sinF)_HLROJO':ASMDM_SF_HLROJO,
         'EntropíaDM(sinF)_HLROJO':entropiaDM_SF_HLROJO,
         'ContrasteDM(sinF)_HLROJO':contrasteDM_SF_HLROJO,
         'Media(glcm)DM(sinF)_HLROJO':mediaglcmDM_SF_HLROJO,
         'Entropia(no glcm)DM(sinF)_HLROJO':entropianoglcmDM_SF_HLROJO,
         'MediaDM(sinF)_HLROJO':mediaDM_SF_HLROJO,
         'ModaDM(sinF)_HLROJO':modaDM_SF_HLROJO,
         'DesviacionDM(sinF)_HLROJO':desviacionDM_SF_HLROJO,
         'CurtosisDM(sinF)_HLROJO':curtosisDM_SF_HLROJO,
         'Energia(no glcm)DM(sinF)_HLROJO':nergianoglcmDM_SF_HLROJO,         
          #RE
         'EnergiaRE(sinF)_HLROJO':energiaRE_SF_HLROJO,
         'HomogeneidadRE(sinF)_HLROJO':homogeneidadRE_SF_HLROJO,
         'CorrelaciónRE(sinF)_HLROJO':correlacionRE_SF_HLROJO,
         'DisimilitudRE(sinF)_HLROJO':disimiRE_SF_HLROJO,
         'ASMRE(sinF)_HLROJO':ASMRE_SF_HLROJO,
         'EntropíaRE(sinF)_HLROJO':entropiaRE_SF_HLROJO,
         'ContrasteRE(sinF)_HLROJO':contrasteRE_SF_HLROJO,
         'Media(glcm)RE(sinF)_HLROJO':mediaglcmRE_SF_HLROJO,
         'Entropia(no glcm)RE(sinF)_HLROJO':entropianoglcmRE_SF_HLROJO,
         'MediaRE(sinF)_HLROJO':mediaRE_SF_HLROJO,
         'ModaRE(sinF)_HLROJO':modaRE_SF_HLROJO,
         'DesviacionRE(sinF)_HLROJO':desviacionRE_SF_HLROJO,
         'CurtosisRE(sinF)_HLROJO':curtosisRE_SF_HLROJO,
         'Energia(no glcm)RE(sinF)_HLROJO':nergianoglcmRE_SF_HLROJO,         
          #NO
         'EnergiaNO(sinF)_HLROJO':energiaNO_SF_HLROJO,
         'HomogeneidadNO(sinF)_HLROJO':homogeneidadNO_SF_HLROJO,
         'CorrelaciónNO(sinF)_HLROJO':correlacionNO_SF_HLROJO,
         'DisimilitudNO(sinF)_HLROJO':disimiNO_SF_HLROJO,
         'ASMNO(sinF)_HLROJO':ASMNO_SF_HLROJO,
         'EntropíaNO(sinF)_HLROJO':entropiaNO_SF_HLROJO,
         'ContrasteNO(sinF)_HLROJO':contrasteNO_SF_HLROJO,
         'Media(glcm)NO(sinF)_HLROJO':mediaglcmNO_SF_HLROJO,
         'Entropia(no glcm)NO(sinF)_HLROJO':entropianoglcmNO_SF_HLROJO,
         'MediaNO(sinF)_HLROJO':mediaNO_SF_HLROJO,
         'ModaNO(sinF)_HLROJO':modaNO_SF_HLROJO,
         'DesviacionNO(sinF)_HLROJO':desviacionNO_SF_HLROJO,
         'CurtosisNO(sinF)_HLROJO':curtosisNO_SF_HLROJO,
         'Energia(no glcm)NO(sinF)_HLROJO':nergianoglcmNO_SF_HLROJO,         
       
         }

datos = pd.DataFrame(datos)
datos.to_excel('DiferentesCaracteristicas.xlsx')        
 