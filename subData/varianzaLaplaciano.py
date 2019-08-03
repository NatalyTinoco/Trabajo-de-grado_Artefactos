# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:50:43 2019

@author: Nataly
"""

from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
from filtromascara import filtro
from agregaceros import agregarceros
import cv2
import numpy as np

def normalizacionMaxMin(img):
    img2 = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img2
from skimage.feature import greycomatrix, greycoprops
import skimage.feature

import glob
entropi=np.zeros((304,2))
import pylab as plt 
import pylab as plt 
from matplotlib import pyplot as plt
i=0
melocoton=0
z=0

for imgfile in glob.glob("*.jpg"):
    ima='./segROI/#5/Z3/'+imgfile
    imaROI=cv2.imread(ima,0)
    imaROI = cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    print(imgfile)
    img=cv2.imread(imgfile)
    img=normalizacionMaxMin(img)
    #HSV=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    #V,H,S=cv2.split(HSV)
    #YUV=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    #Y,U,V=cv2.split(YUV)
   
    V=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    V=V*imaROI
    #laplaciano=1/6*np.array([[0,1,0], [1,-4,1],[0,1,0]])
    lxx=1/6*np.array([[0,0,1], [0,-2,0],[1,0,0]])
    lyy=1/6*np.array([[1,0,0], [0,-2,0],[0,0,1]])
    
    #tipomas=3
    #Vnu,tama,tamanu=agregarceros(V,tipomas)
    #la=filtro(Vnu,tama,tamanu,tipomas,laplaciano,V)
    lx2=cv2.filter2D(V, -1, lxx)
    ly2=cv2.filter2D(V, -1, lyy)
    lx=np.array([-1,2,-1])
    ly=np.transpose(lx)
    lxd=cv2.filter2D(V, -1, lx)
    lyd=cv2.filter2D(V, -1, ly)
    la=lxd+lyd+lx2+ly2
    
    
    #plt.imshow(la,'Greys')
    #plt.show()
    La=0
    f,c=la.shape
    suma=0
    for x in range(f):
        for y in range (c):
            suma=suma+abs(la[x,y])
            #print(suma)
    #print(suma)
    La=(1/f*c)*suma
    varLa=0
    for x in range(f):
        for y in range (c):
            varLa=(varLa)+(la[x,y]-La)**(2)
            #print(suma)
    #print(varLa)
    filetxt=imgfile[0:len(imgfile)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, img.shape)  
    re=0
    dm=0
    
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            dm=dm+1
        else:
            re=re+1
    if re>0 and dm==0:
        entropi[melocoton,1]=varLa
        print('re')
        melocoton=melocoton+1
    else:
        print('dm')
        entropi[z,0]=varLa
        z=z+1
    print(melocoton,z)
    i=i+1 

plt.plot(entropi[0:z,0],'o')
plt.plot(entropi[0:melocoton,1],'*')
plt.show()
    
"""
    filetxt=imgfile[0:len(imgfile)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, img.shape)  
    re=0
    dm=0
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            dm=dm+1
        else:
            re=re+1
    if re>0 and dm==0:
        contrast[i,1]=contraste
        energi[i,1]=energia
        homogeneida[i,1]=homogeneidad
        correlacio[i,1]=correlacion
        disim[i,1]=disimi
        AS[i,1]=ASM
        entropi[i,1]=entropia
        print('re')
    else:
        contrast[i,0]=contraste
        energi[i,0]=energia
        homogeneida[i,0]=homogeneidad
        correlacio[i,0]=correlacion
        disim[i,0]=disimi
        AS[i,0]=ASM
        entropi[i,0]=entropia
        print('dm')
        
    i=i+1 
    

table=[contrast,energi,homogeneida,correlacio,disim,AS,entropi]
tabla=['Contraste','Energía','Homogeneidad','Correlación','Disimilitud','ASM','Entropía']

a=0
b=1
h=1

for i in range (len(table)*3):
    print(i)
    print(a,b)
    #with PdfPages('canalV_SinFourier.pdf') as pdf:
    f=plt.figure()
    plt.scatter(table[a][:,0],table[b][:,0],c='blue', label='DM')
    plt.scatter(table[a][:,1],table[b][:,1],c='red', label='RE')            
    plt.title('GLCM')
    plt.xlabel(tabla[a])
    plt.ylabel(tabla[b])
    plt.legend(loc='top_left')
    plt.show()
    f.savefig('./diagramasDeDis/canalV_ConFourier/'+str(i))
    #f.savefig("canalV_SinFourier.pdf", bbox_inches='tight')
    plt.close()
    b=b+1
    if b==len(tabla):
        h=h+1
        b=h
        a=a+1
"""