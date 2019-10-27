"""
Created on Fri Aug 16 22:44:45 2019

@author: Nataly
"""
import sys
sys.path.insert(1, 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/funciones')
from test_todoRE import test_all_RE
from test_todoDM import test_all_DM
#from readimg import read_img
import cv2
import numpy as np
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
import math

test = []
prueb = []
for image in glob.glob('*.jpg'):    
#    image='00039.jpg'
    im = cv2.imread(image)
    filetxt='C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/Prueba/Todas/'+image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    dm=0
    for b in boxes_abs:
            cls, x1, y1, x2, y2 = b
            print(cls)
            if cls == 3:
                dm=dm+1
            if cls==0:
                re=re+1
                
    if re > 0 and dm > 0:
        test.append(3)
#        print('2')
    if re>0 and dm==0:
         test.append(1)
#         print('1')
    if re==0 and dm>0:
         test.append(0)
         cv2.imwrite('./si/'+image,im)
#         print('0')
    if re==0 and dm==0:
         test.append(2)  
         print('2')
#    resul, original_2,imDU_2,umbrImage,original_3=test_all_RE(image)
#    resuldm,originaldm_2,imDRdm_2,original_3=test_all_DM(image,original_3)
#    
#    grupo0 = [i for i,x in enumerate(resul) if x == 0]
#    grupo1 = [i for i,x in enumerate(resul) if x == 1]
#    grupodm0 = [i for i,x in enumerate(resuldm) if x == 0]
#    grupodm1 = [i for i,x in enumerate(resuldm) if x == 1]
##    if len(grupo1)>0 and len(grupodm0)==0:
##        prueb.append(1)
#    if len(grupodm0)>0 and len(grupo1)==0:
#        prueb.append(0)
#        cv2.imwrite('./dm/'+image,im)
#    if  len(grupodm0)>0 and len(grupo1)>0 :
#        prueb.append(3)  
#        cv2.imwrite('./dm/'+image,im)
#    if len(grupo0)>0 and len(grupodm1)>0 and len(grupo1)==0 and len(grupodm0)==0 or math.isnan(np.mean(resul))==True and len(grupodm1)>0 and len(grupodm0)==0:
#        prueb.append(2)

import pandas as pd  
datos = {'test': test,}
#         'prueba':prueb}

datos = pd.DataFrame(datos)

datos.to_excel('evaluaciontest.xlsx')      