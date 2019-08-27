# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:44:06 2019

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
from scipy.linalg import hadamard
from numpy.linalg import norm
from scipy.stats import linregress
import math
import pylab as py

import sys
sys.path.insert(1,'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from brilloContraste import contraste, brillo
from yolovoc import yolo2voc
from readboxes import read_boxes
from rOI import ROI
from ventaneo import ventaneoo
from agregaceros import agregarceros
from filtromedianaa import filtromediana
import pywt
import pywt.data
from numpy import linalg as LA
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
from skimage.measure import compare_ssim as ssim
from skimage.filters.rank import median
from skimage.morphology import disk

def Fourier(inA):
    f = np.fft.fft2(inA)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    return fourier 
def DFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape 
    crow,ccol = int(rows/2) , int(cols/2)
    print(rows,cols,crow,ccol)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back
    
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

def azimuthalAverage(image, center=None):
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def SingularValueFeature(A):
    #import numpy.linalg as svd 
    k,k1=A.shape
    U,s,V=np.linalg.svd(A,full_matrices=False)
    #print(U.shape,s.shape,V.shape)
    reconst_matrix=np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return  reconst_matrix,s
def HOP(image):
    import matplotlib.pyplot as plt
    from skimage.feature import hog
    from skimage import data, exposure
    #image = data.astronaut()
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled

def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def L2norm2images(aa,bb,pp,LH,LL):
    su=0
    sur=0
    p=pp
    for f in range(aa):
        for c in range(bb):
            sur=abs(LH[f,c]**(p))+sur
            su=abs(LL[f,c]**(p))+su
    sur=sur**(1/p)
    su=su**(1/p)    
    beta1=sur/su
    dif=su-sur
    return beta1,dif

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err    

contrastTF=[]
energiTF=[]
homogeneiTF=[]
correlaciTF=[]
disiTF=[]
ASTF=[]
entropTF=[]

contrastDFT=[]
energiDFT=[]
homogeneiDFT=[]
correlaciDFT=[]
disiDFT=[]
ASDFT=[]
entropDFT=[]  

contrast=[]
energi=[]
homogenei=[]
correlaci=[]
disi=[]
AS=[]
entrop=[]  


l2waveletLL=[]
l2waveletLH=[]
l2waveletHL=[]
l2waveletHH=[]
l2altas=[]
l2bajas=[]
difwaveletLH=[]
difwaveletHL=[]
difwaveletHH=[]
mediaLL=[]
mediaLH=[]
mediaHL=[]
mediaHH=[]
mediaaltas=[]
mediabajas=[]
varLL=[]
varLH=[]
varHL=[]
varHH=[]
varbajas=[]
varaltas=[]
entropiaLL=[]
entropiaLH=[]
entropiaHL=[]
entropiaHH=[]
entropiabajas=[]
entropiaaltas=[]


beta=[]
sumas=[]
media=[]
mediana=[]
destan=[]
var=[]
l2=[]


varlaplacianargb=[]
varlaplacianaV=[]

betaHOP=[]
sumasHOP=[]
mediaHOP=[]
medianaHOP=[]
destanHOP=[]
varHOP=[]
l2HOP=[]


valorintensidadHFOUR=[]
valorpicoHFOUR=[]
sumasHFOUR=[]
mediaHFOUR=[]
medianaHFOUR=[]
destanHFOUR=[]
varHFOUR=[]
asimetriaHFOUR=[]
kurtosHFOUR=[]
betadmHFOUR=[]  



valorintensidadHDFT=[]
valorpicoHDFT=[]
sumasHDFT=[]
mediaHDFT=[]
medianaHDFT=[]
destanHDFT=[]
varHDFT=[]  
asimetriaHDFT=[]
kurtosHDFT=[]
betadmHDFT=[]  




valorintensidadEDFT=[]
#valorpicoEDFT=[]
sumasEDFT=[]
mediaEDFT=[]
medianaEDFT=[]
destanEDFT=[]
varEDFT=[]
betadmEDFT=[]   
asimetriaEDFT=[]
kurtosEDFT=[]
pendientereEDFT=[]
maxderivadaEDFT=[]



betaDWHT=[]
DiferenciaDWHT=[]
errorCuadraticoDWHT=[]
sSIMDWHT=[]



betanor=[]
Diferencianor=[]
errorCuadraticonor=[]
sSIMnor=[]


brilloop2= []
brillomedia = []
contras = []
brillomediana=[]

i=0
import matplotlib.pyplot as plt
from skimage import exposure
def contraststretching(img):
    #contrast Stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

for imgfile in glob.glob("*.jpg"):
    im = cv2.imread(imgfile)
    aa,bb,c = im.shape    
#    croppedrgb=im
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#    im=contraststretching(im)
    croppedrgb=im.copy()
    
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
#    croppedrgb=HSV.copy()
#    cropped=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    H,S,cropped=cv2.split(HSV)
#    R,G,cropped=cv2.split(im)
#    plt.imshow(cropped,'Greys')
#    plt.show()
    """TF"""
    cropFou=Fourier(cropped)
    g,contrastel,energia,homogeneidad, correlacion, disimi, ASM,entropia=GLCM(cropFou)
    contrastTF.append(contrastel)
    energiTF.append(energia)
    homogeneiTF.append(homogeneidad)
    correlaciTF.append(correlacion)
    disiTF.append(disimi)
    ASTF.append(ASM)
    entropTF.append(entropia)
    """DF"""
    cropFou2=DFT(cropped)
    cropFou2= cropFou2.astype(np.uint8)
#    plt.imshow(cropFou2)
#    plt.show()
    g2,contraste2,energia2,homogeneidad2, correlacion2, disimi2, ASM2,entropia2=GLCM(cropFou2)
    contrastDFT.append(contraste2)
    energiDFT.append(energia2)
    homogeneiDFT.append(homogeneidad2)
    correlaciDFT.append(correlacion2)
    disiDFT.append(disimi2)
    ASDFT.append(ASM2)
    entropDFT.append(entropia2)
    """ Sin"""
    cropSinFou=cropped.copy()
    g3,contraste3,energia3,homogeneidad3, correlacion3, disimi3, ASM3,entropia3=GLCM(cropSinFou)
    contrast.append(contraste3)
    energi.append(energia3)
    homogenei.append(homogeneidad3)
    correlaci.append(correlacion3)
    disi.append(disimi3)
    AS.append(ASM3)
    entrop.append(entropia3)
    """Wavelet"""
    coeffs2 = pywt.dwt2(cropped, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    bajasfrec=LL.copy()
    altasfrec=LH+HL+HH
    l2waveletLL.append(LA.norm(LL))
    l2waveletLH.append(LA.norm(LH))
    l2waveletHL.append(LA.norm(HL))
    l2waveletHH.append(LA.norm(HH))
    l2altas.append(LA.norm(altasfrec))
    l2bajas.append(LA.norm(bajasfrec))
    pp=2
    aal,bbl=LL.shape
    beta1,dif=L2norm2images(aal,bbl,pp,LH,LL)
    difwaveletLH.append(dif)
    beta2,dif2=L2norm2images(aal,bbl,pp,HL,LL)
    difwaveletHL.append(dif2)
    beta3,dif3=L2norm2images(aal,bbl,pp,HH,LL)
    difwaveletHH.append(dif3)
    mediaLL.append(np.mean(LL))
    mediaLH.append(np.mean(LH))
    mediaHL.append(np.mean(HL))
    mediaHH.append(np.mean(HH))
    mediaaltas.append(np.mean(altasfrec))
    mediabajas.append(np.mean(bajasfrec))
    varLL.append(np.var(LL))
    varLH.append(np.var(LH))
    varHL.append(np.var(HL))
    varHH.append(np.var(HH))
    varbajas.append(np.var(bajasfrec))
    varaltas.append(np.var(altasfrec))
    entropiaLL.append(skimage.measure.shannon_entropy(LL))
    entropiaLH.append(skimage.measure.shannon_entropy(LH))
    entropiaHL.append(skimage.measure.shannon_entropy(HL))
    entropiaHH.append(skimage.measure.shannon_entropy(HH))
    entropiaaltas.append(skimage.measure.shannon_entropy(altasfrec))
    entropiabajas.append(skimage.measure.shannon_entropy(bajasfrec))
    
    """SVD"""
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
    l2.append(LA.norm(T))
    """ Varianza Lap"""
    varlaplacianargb.append(cv2.Laplacian(croppedrgb, cv2.CV_64F).var())
    varlaplacianaV.append(cv2.Laplacian(cropped, cv2.CV_64F).var())
    """ HOP"""
    gradiente=HOP(croppedrgb)
    TT=gradiente.tolist() 
    uu=np.max(TT)
    suma=np.asarray(TT)
    suma=suma.flatten()
    sumasHOP.append(sum(suma))
    mediaHOP.append(np.mean(gradiente))
    medianaHOP.append(np.median(gradiente))
    destanHOP.append(np.std(gradiente))
    varHOP.append(np.var(gradiente))
    l2HOP.append(LA.norm(gradiente))
    
    """ Histograma FOURIER"""
    #FT
    croppedHOUR=Fourier(cropped)
    hist = cv2.calcHist([croppedHOUR],[0],None,[256],[0,255])
    hisa=hist.copy()
    hist=hist.tolist() 
    uHOUR=np.max(hist)
    hi=hist.index( uHOUR)
    valorintensidadHFOUR.append(hi)
    valorpicoHFOUR.append(hisa[hi])
    if hi==0:
       betadmHFOUR.append(hisa[hi]/sum(hisa))
    else:
        betadmHFOUR.append(sum(hisa[0:hi])/sum(hisa))
    asimetriaHFOUR.append(skew(hisa))
    kurtosHFOUR.append(kurtosis(hisa))
    sumasHFOUR.append(sum(hisa))
    mediaHFOUR.append(np.mean(hisa))
    medianaHFOUR.append(np.median(hisa))
    destanHFOUR.append(np.std(hisa))
    varHFOUR.append(np.var(hisa))
    #dft
    croppedHDFT=DFT(cropped)
    croppedHDFT=croppedHDFT.astype(np.uint8)
    histDFT = cv2.calcHist([croppedHDFT],[0],None,[256],[0,255])
    hisaDFT=histDFT.copy()
    histDFT=histDFT.tolist() 
    uDFT=np.max(histDFT)
    hiDFT=histDFT.index(uDFT)
    valorintensidadHDFT.append(hiDFT)
    valorpicoHDFT.append(hisaDFT[hiDFT])
    if hiDFT==0:
       betadmHDFT.append(hisaDFT[hiDFT]/sum(hisaDFT))
    else:
        betadmHDFT.append(sum(hisaDFT[0:hiDFT])/sum(hisaDFT))
    asimetriaHDFT.append(skew(hisaDFT))
    kurtosHDFT.append(kurtosis(hisaDFT))
    sumasHDFT.append(sum(hisaDFT))
    mediaHDFT.append(np.mean(hisaDFT))
    medianaHDFT.append(np.median(hisaDFT))
    destanHDFT.append(np.std(hisaDFT))
    varHDFT.append(np.var(hisaDFT))
    
    """ Especro de poencias"""
    img_backc=DFT(cropped)
    jc=azimuthalAverage(img_backc)
    mediaEDFT.append(np.mean(jc))
    medianaEDFT.append(np.median(jc))
    destanEDFT.append(np.std(jc))
    varEDFT.append(np.var(jc))
    asimetriaEDFT.append(skew(jc))
    kurtosEDFT.append(kurtosis(jc))
    gradientec=np.gradient(jc)
    alfapma=np.max(gradientec)  
    xsc=np.arange(len(jc))
    slopec = linregress(xsc, jc)[0]  # slope in units of y / x
    slope_anglec = math.atan(slopec)  # slope angle in radians
    alfap = math.degrees(slope_anglec) 
    pendientereEDFT.append(alfap)
    maxderivadaEDFT.append(alfapma)
    

    """ DWHTs"""
    if aa>bb:
        poten=math.log(aa,2)
        poten=int(poten)
    else:
        poten=math.log(bb,2)
        poten=int(poten)
    tamañoa1A=2**poten
    tamañoa1B=2**poten
    
    HN=hadamard(tamañoa1A, dtype=complex).real
    HT=np.transpose(HN)
    
    
    aaa = cv2.resize(cropped,( tamañoa1A, tamañoa1B))
    WHT=HN*aaa*HT
    
    
#    bbb = cv2.GaussianBlur(aaa,(5,5),2.5)
    bbb=median(aaa, disk(20))
    WHTr=HT*bbb*HT
    
#    plt.imshow(WHT,'Greys')
#    plt.show()
#    plt.imshow(WHTr,'Greys')
#    plt.show()
    
    p1=0.76
    beta1W,difW=L2norm2images(tamañoa1A,tamañoa1B,p1,WHTr,WHT)
    betaDWHT.append(beta1W)
    DiferenciaDWHT.append(difW)    
    errorCuadraticoDWHT.append(mse(WHTr,WHT))
    sSIMDWHT.append(ssim(WHTr,WHT))
    
    
    """ sin DWTs"""
    aa4=cropped.copy()
#    bb = cv2.GaussianBlur(aa,(5,5),2.5)
    bb4=median(aa4, disk(20))
    beta1nor,difnor=L2norm2images(aa,bb,p1,bb4,aa4)
    betanor.append(beta1nor)
    Diferencianor.append(difnor)
    errorCuadraticonor.append(mse(bb4,aa4))
    sSIMnor.append(ssim(bb4,aa4))
    
    """ brillos"""
    R,G,B=cv2.split(croppedrgb)
    brillon=np.sqrt(0.241*R**2+0.691*G**2+0.068*B**2)/(aa*bb)
    brillomedia.append(np.mean(brillon))
    brillomediana.append(np.median(brillon))
    contras.append(contraste(cropped))
    print(brillo(cropped))
    
    i=i+1
    print('IMAGEN',i)

import pandas as pd    
datos = {'ContrasteTF':contrastTF,
         'Energia':energiTF,
         'Homogeneidad':homogeneiTF,
         'Correlación':correlaciTF,
         'Disimilitud':disiTF,
         'ASM':ASTF,
         'Entropia':entropTF,
         'ContrasteDFT':contrastDFT,
         'EnergiaDFT':energiDFT,
         'HomogeneidadDFT':homogeneiDFT,
         'CorrelaciónDFT':correlaciDFT,
         'DisimilitudDFT':disiDFT,
         'ASMDFT':ASDFT,
         'EntropiaDFT':entropDFT,
         'ContrasteSF':contrast,
         'EnergiaSF':energi,
         'HomogeneidadSF':homogenei,
         'CorrelaciónSF':correlaci,
         'DisimilitudSF':disi,
         'ASMSF':AS,
         'EntropiaSF':entrop,
         'l2waveletLL':l2waveletLL,
         'l2waveletLH':l2waveletLH,
         'l2waveletHL':l2waveletHL,
         'l2waveletHH':l2waveletHH,
         'l2altas':l2altas,
         'l2bajas':l2bajas,
         'difwaveletLH':difwaveletLH,
         'difwaveletHL':difwaveletHL,
         'difwaveletHH':difwaveletHH,
         'mediaLL':mediaLL,
         'mediaLH':mediaLH,
         'mediaHL':mediaHL,
         'mediaHH':mediaHH,
         'mediaaltas':mediaaltas,
         'mediabajas':mediabajas,
         'varLL':varLL,
         'varLH':varLH,
         'varHL':varHL,
         'varHH':varHH,
         'varbajas':varbajas,
         'varaltas':varaltas,
         'entropiaLL':entropiaLL,
         'entropiaLH':entropiaLH,
         'entropiaHL':entropiaHL,
         'entropiaHH':entropiaHH,
         'entropiabajas':entropiabajas,
         'entropiaaltas':entropiaaltas,
         'beta':beta,
         'sumas':sumas,
         'media':media,
         'mediana':mediana,
         'destan':destan,
         'var':var,
         'l2':l2,
         'varlaplacianargb':varlaplacianargb,
         'varlaplacianaV':varlaplacianaV,
#         'betaHOP':betaHOP,
         'sumasHOP':sumasHOP,
         'mediaHOP':mediaHOP,
         'medianaHOP':medianaHOP,
         'destanHOP':destanHOP,
         'varHOP':varHOP,
         'l2HOP':l2HOP,
         'valorintensidadHFOUR':valorintensidadHFOUR,
         'valorpicoHFOUR':valorpicoHFOUR,
         'sumasHFOUR':sumasHFOUR,
         'mediaHFOUR':mediaHFOUR,
         'medianaHFOUR':medianaHFOUR,
         'destanHFOUR':destanHFOUR,
         'varHFOUR':varHFOUR,
         'asimetriaHFOUR':asimetriaHFOUR,
         'kurtosHFOUR':kurtosHFOUR,
         'betadmHFOUR':betadmHFOUR,
         'valorintensidadHDFT':valorintensidadHDFT,
         'valorpicoHDFT':valorpicoHDFT,
         'sumasHDFT':sumasHDFT,
         'mediaHDFT':mediaHDFT,
         'medianaHDFT':medianaHDFT,
         'destanHDFT':destanHDFT,
         'varHDFT':varHDFT,
         'asimetriaHDFT':asimetriaHDFT,
         'kurtosHDFT':kurtosHDFT,
         'betadmHDFT':betadmHDFT,
#         'valorintensidadEDFT':valorintensidadEDFT,
#         'valorpicoEDFT':valorpicoEDFT,
#         'sumasEDFT':sumasEDFT,
         'mediaEDFT':mediaEDFT,
         'medianaEDFT':medianaEDFT,
         'destanEDFT':destanEDFT,
         'varEDFT':varEDFT,
#         'betadmEDFT': betadmEDFT,
         'asimetriaEDFT':asimetriaEDFT,
         'kurtosEDFT':kurtosEDFT,
         'pendientereEDFT':pendientereEDFT,
         'maxderivadaEDFT':maxderivadaEDFT,
         
         'betaDWHT':betaDWHT,
         'DiferenciaDWHT':DiferenciaDWHT,
         'errorCuadraticoDWHT':errorCuadraticoDWHT,
         'sSIMDWHT':sSIMDWHT,
         'betaN':betanor,
         'DiferenciaN':Diferencianor,
         'errorCuadraticoN':errorCuadraticonor,
         'sSIMN':sSIMnor,
         'brillomedia':brillomedia,
         'contras':contras,
         'brillomediana':brillomediana}

datos = pd.DataFrame(datos)
datos.to_excel('Caracateristicas_NO_HS(V)_50x50.xlsx') 
#datos.to_excel('Caracateristicas_NO_H(S)V_50x50.xlsx') 
#datos.to_excel('Caracateristicas_NO_RG(B)_50x50.xlsx') 
#datos.to_excel('Caracateristicas_NO_(HSV)Grey_50x50.xlsx') 
#datos.to_excel('Caracateristicas_NO_Estiramiento_H(S)V_50x50.xlsx') 
#datos.to_excel('Caracateristicas_NO_Estiramiento_HS(V)_50x50.xlsx') 