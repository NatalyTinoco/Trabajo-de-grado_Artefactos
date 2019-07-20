# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:50:31 2019

@author: Nataly
"""

"""
file='Lap_01004'
file='DB58_colitis_full_00479'
file='DB58_colitis_full_03038'
#file='gastro_00377'
#file='CT56_colitis_02461'
file='Flu_01025'
file='00075'
#file='WL_00485'
file='00119_batch2'
imgfile=file+'.jpg'
filetxt=file+'.txt'
    
bboxfile=filetxt
#classfile=filetxt
#classes = read_obj_names(classfile)
img = read_img(imgfile)
#img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
boxes = read_boxes(bboxfile)
# convert boxes from (x,y,w,h) to (x1,y1,x2,y2) format for plotting
boxes_abs = yolo2voc(boxes, img.shape)


#import seaborn as sns 
#color_pal = sns.color_palette('hls', n_colors = 7)
#r=0
for b in boxes_abs:
    cls, x1, y1, x2, y2 = b
    if cls == 3:
"""
def comparacionhistDM(img,x1, y1, x2, y2):
    #def comparacionhistRE(imgfile,filetxt):
    from readimg import read_img #leer imagines ### img=read_img(imgfile)##
    from readboxes import read_boxes #leer bbox ## boxes=read_boxes(txtfile) ##
    from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
    from plotboxes import plot_boxes #graficar bbox ## plot_boxes(ax, boxes)##
    from readobjnames import read_obj_names  ## classes = read_obj_names(classfile) ##
    
    import cv2
    import pylab as plt 
    from matplotlib import pyplot as plt
    import numpy as np
    from numpy import asarray
    from PIL import Image
    from math import floor
    import pandas as pd
    import statistics as stats
    artefacto=img[int(y1):int(y2),int(x1):int(x2),:]
    hista = cv2.calcHist([artefacto], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    plt.imshow(artefacto)
    plt.show()
    
    ta=img.shape
    ta=list(ta)
    tavta=artefacto.shape
    tavta=list(tavta)
    factor=2
     
    if tavta[0]>=ta[0] & tavta[1]>=ta[1]:
        vecindad=img[int(y1):int(y2),int(x1):int(x2),:]
    elif tavta[0]>=ta[0]:     
        vecindad=img[int(y1):int(y2),int(x1-factor):int(x2+factor),:]
    elif tavta[1]>=ta[1]: 
        vecindad=img[int(y1-factor):int(y2+factor),int(x1):int(x2),:]
    else:
         vecindad=img[int(y1-factor):int(y2+factor),int(x1-factor):int(x2+factor),:]   
    plt.imshow(vecindad)
    plt.show()
    
    tav=vecindad.shape
    tav=list(tav)
    mascara=np.ones([tav[0],tav[1]])
    for y in range(int(y1),int(y2)+1):
        for x in range(int(x1),int(x2)+1):
            #print(y,x)
            mascara[y,x]=0
   # mascara=mascara[int(y1-factor):int(y2+factor),int(x1-factor):int(x2+factor)]
    mascara=mascara.astype(np.uint8)
    plt.imshow(mascara,'Greys')
    plt.show()
    

    for z in range((3)):
        vecindad[:,:,z]=(vecindad[:,:,z]*(mascara))
    
    histb = cv2.calcHist([vecindad], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    plt.imshow(vecindad)
    plt.show()
    #histvecindad=histograma3(vecindad)
     ### Calculo de medidas
    from scipy.spatial import distance as dist
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2    
    #Correlación
    correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
    #Distancia Bhattacharyya
    Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA)
    return artefacto,vecindad,hista,histb,correlacion,Bhattacharyya
            
"""
file='Flu_01025'  
#file='gastro_00377'  
file='Lap_01004'
file='DB58_colitis_full_00479'
#file='DB58_colitis_full_03038'
imgfile=file+'.jpg'
filetxt=file+'.txt'
artefacto,vecindad,hista,histb,correlacion,Bhattacharyya=comparacionhistRE(imgfile,filetxt)
from matplotlib import pyplot as plt
import cv2
artefacto=cv2.cvtColor(artefacto,cv2.COLOR_RGB2BGR)
vecindad=cv2.cvtColor(vecindad,cv2.COLOR_RGB2BGR)
plt.imshow(artefacto)
plt.show()
plt.imshow(vecindad)
plt.show()
print('Correlación=', correlacion)
print('Distancia Bhattacharyya=', Bhattacharyya)
"""
