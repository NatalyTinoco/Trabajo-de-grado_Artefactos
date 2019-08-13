# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:43:27 2019

@author: Nataly
"""

import pywt
import pywt.data
from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
import cv2
import numpy as np
from rOI import ROI
def normalizacionMaxMin(img):
    img2 = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img2
from skimage.feature import greycomatrix, greycoprops
import skimage.feature

import glob
contrast=np.zeros((304,2))
energi=np.zeros((304,2))
homogeneida=np.zeros((304,2))
correlacio=np.zeros((304,2))
disim=np.zeros((304,2))
AS=np.zeros((304,2))
entropi=np.zeros((304,2))
import pylab as plt 
import pylab as plt 
from matplotlib import pyplot as plt
i=0
melocoton=0
z=0
for imgfile in glob.glob("*.jpg"):
    img=cv2.imread(imgfile)
    img=normalizacionMaxMin(img)
    
    imaROI=ROI(img)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    print(imgfile)
    
    HSV=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
   # YUV=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
   # Y,U,V=cv2.split(YUV)
    
    for k in range(3):
        img[:,:,k]=img[:,:,k]*imaROI
    V=V*imaROI
    
    
#
#    g = skimage.feature.greycomatrix(LL, [1], [0], levels=a+1, symmetric=False, normed=True)    
#    contraste=skimage.feature.greycoprops(g, 'contrast')[0][0]
#    energia=skimage.feature.greycoprops(g, 'energy')[0][0]
#    homogeneidad=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
#    correlacion=skimage.feature.greycoprops(g, 'correlation')[0][0]
#    disimi= greycoprops(g, 'dissimilarity') 
#    ASM= greycoprops(g, 'ASM')
#    entropia=skimage.measure.shannon_entropy(g)
    #entropia=np.var(HL)
    filetxt=imgfile[0:len(imgfile)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, img.shape)  
    re=0
    dm=0
    DEMO=0
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            dm=dm+1
        if cls==0:
            re=re+1
        if cls==2:
            DEMO=DEMO+1
    if re>0 and dm==0 and DEMO==0:
#        contrast[melocoton,1]=contraste
#        energi[melocoton,1]=energia
#        homogeneida[melocoton,1]=homogeneidad
#        correlacio[melocoton,1]=correlacion
#        disim[melocoton,1]=disimi
#        AS[melocoton,1]=ASM
#        entropi[melocoton,1]=entropia
#         plt.imshow(img)
#         plt.show()
#         titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
#         coeffs2 = pywt.dwt2(img, 'bior1.3')
#         LL, (LH, HL, HH) = coeffs2
#         #    """
#        
#         fig = plt.figure(figsize=(12, 3))
#         for i, a in enumerate([LL, LH, HL, HH]):
#             ax = fig.add_subplot(1, 4, i + 1)
#             ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#             ax.set_title(titles[i], fontsize=10)
#             ax.set_xticks([])
#             ax.set_yticks([])
#        
#         fig.tight_layout()
#         plt.show()
#         entropia=skimage.measure.shannon_entropy(HL)
#         var=np.var(HL)
#         print('rgb',entropia,var)
#         
#         titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
#         coeffs2 = pywt.dwt2(V, 'bior1.3')
#         LL, (LH, HL, HH) = coeffs2
#         #    """
#        
#         fig = plt.figure(figsize=(12, 3))
#         for i, a in enumerate([LL, LH, HL, HH]):
#             ax = fig.add_subplot(1, 4, i + 1)
#             ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#             ax.set_title(titles[i], fontsize=10)
#             ax.set_xticks([])
#             ax.set_yticks([])
#         plt.show()
##         #"""
##         #LH=LH.astype(np.uint8)
##         #a=int(np.max(LH))
#         entropia=skimage.measure.shannon_entropy(V)
#         var=np.var(V)
#         print('V',entropia,var)
#         LL=LL.astype(np.uint8)
#         a=int(np.max(LL))
         print('re')
       
        #print(entropia)
         #print(melocoton)
         melocoton=melocoton+1
        
    if dm>0:
         print('dm')
         plt.imshow(img)
         plt.show()
         titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
         coeffs2 = pywt.dwt2(img, 'bior1.3')
         LL, (LH, HL, HH) = coeffs2
         #    """
        
         fig = plt.figure(figsize=(12, 3))
         for i, a in enumerate([LL, LH, HL, HH]):
             ax = fig.add_subplot(1, 4, i + 1)
             ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
             ax.set_title(titles[i], fontsize=10)
             ax.set_xticks([])
             ax.set_yticks([])
        
         fig.tight_layout()
         plt.show()
         entropia=skimage.measure.shannon_entropy(HL)
         var=np.var(HL)
         print('rgb',entropia,var)
         
         titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
         coeffs2 = pywt.dwt2(V, 'bior1.3')
         LL, (LH, HL, HH) = coeffs2
         #    """
        
         fig = plt.figure(figsize=(12, 3))
         for i, a in enumerate([LL, LH, HL, HH]):
             ax = fig.add_subplot(1, 4, i + 1)
             ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
             ax.set_title(titles[i], fontsize=10)
             ax.set_xticks([])
             ax.set_yticks([])
#         plt.imshow()
         plt.show()
#         #"""
#         #LH=LH.astype(np.uint8)
#         #a=int(np.max(LH))
#         plt.imshow(V)
#         plt.show()
#         entropia=skimage.measure.shannon_entropy(V)
#         var=np.var(V)
#         print('V',entropia,var)
#         LL=LL.astype(np.uint8)
#         a=int(np.max(LL))
#        print(entropia)
#        contrast[z,0]=contraste
#        energi[z,0]=energia
#        homogeneida[z,0]=homogeneidad
#        correlacio[z,0]=correlacion
#        disim[z,0]=disimi
#        AS[z,0]=ASM
#        entropi[z,0]=entropia
         z=z+1
         print(z)
    if dm==0 and re==0 and DEMO==0:
        print('otro')
#         plt.imshow(img)
#         plt.show()
#         titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
#         coeffs2 = pywt.dwt2(img, 'bior1.3')
#         LL, (LH, HL, HH) = coeffs2
#         #    """
#        
#         fig = plt.figure(figsize=(12, 3))
#         for i, a in enumerate([LL, LH, HL, HH]):
#             ax = fig.add_subplot(1, 4, i + 1)
#             ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#             ax.set_title(titles[i], fontsize=10)
#             ax.set_xticks([])
#             ax.set_yticks([])
#        
#         fig.tight_layout()
#         plt.show()
#         entropia=skimage.measure.shannon_entropy(HL)
#         var=np.var(HL)
#         print('rgb',entropia,var)
#         
#         titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
#         coeffs2 = pywt.dwt2(V, 'bior1.3')
#         LL, (LH, HL, HH) = coeffs2
#         #    """
#        
#         fig = plt.figure(figsize=(12, 3))
#         for i, a in enumerate([LL, LH, HL, HH]):
#             ax = fig.add_subplot(1, 4, i + 1)
#             ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#             ax.set_title(titles[i], fontsize=10)
#             ax.set_xticks([])
#             ax.set_yticks([])
#         plt.show()
        
    print(melocoton,z)
    i=i+1 
"""
plt.plot(entropi[0:z,0],'o')
plt.plot(entropi[0:melocoton,1],'*')
plt.show()
plt.plot(AS[0:z,0],'o')
plt.plot(AS[0:melocoton,1],'*')
plt.show()    
#cor=np.corrcoef(contrast[:,0],contrast[:,1])
plt.scatter(entropi[0:z,0],AS[0:z,0],c='blue', label='DM')
plt.scatter(entropi[0:z,1],AS[0:z,1],c='red', label='RE')            
plt.title('GLCM')
plt.xlabel('Entropía')
plt.ylabel('Varianza')
plt.legend(loc='top_left')
plt.show() 
#"""
#"""
#table=[contrast,energi,homogeneida,correlacio,disim,AS,entropi]
#tabla=['Contraste','Energía','Homogeneidad','Correlación','Disimilitud','ASM','Entropía']
#
#a=0
#b=1
#h=1
#
#for i in range (len(table)*3):
#    print(i)
#    print(a,b)
#    #with PdfPages('canalV_SinFourier.pdf') as pdf:
#    f=plt.figure()
#    plt.scatter(table[a][0:z,0],table[b][0:z,0],c='blue', label='DM')
#    plt.scatter(table[a][0:melocoton,1],table[b][0:melocoton,1],c='red', label='RE')            
#    plt.title('GLCM')
#    plt.xlabel(tabla[a])
#    plt.ylabel(tabla[b])
#    plt.legend(loc='top_left')
#    plt.show()
#    f.savefig('./diagramasDeDis/otra/'+str(i))
#    #f.savefig("canalV_SinFourier.pdf", bbox_inches='tight')
#    plt.close()
#    b=b+1
#    if b==len(tabla):
#        h=h+1
#        b=h
#        a=a+1
#        
#table=[contrast,energi,homogeneida,correlacio,disim,AS,entropi]
#tabla=['Contraste','Energía','Homogeneidad','Correlación','Disimilitud','ASM','Entropía']
#
#a=0
#b=1
#h=1
#
#for i in range (len(table)):
#    f=plt.figure()
#    plt.ion()
#    plt.boxplot([table[i][0:z,0],table[i][0:melocoton,1]], sym = 'ko', whis = 1.5)  # El valor por defecto para los bigotes es 1.5*IQR pero lo escribimos explícitamente
#    plt.xticks([1,2], ['DM','RE'], size = 'large', color = 'k')  # Colocamos las etiquetas para cada distribución
#    plt.ylabel(tabla[i])
#    plt.show()
#    f.savefig('./diagramasDeDis/otra/'+'boxplot'+str(i))
#    plt.close()
#

        