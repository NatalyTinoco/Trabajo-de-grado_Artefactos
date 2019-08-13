# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:32:22 2019

@author: Nataly
"""
def comparacionhistRE(img,img2,segmenta):
    #from readimg import read_img #leer imagines ### img=read_img(imgfile)##
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
    from scipy.spatial import distance
    import statistics
    imgseg=cv2.imread(segmenta,0)
    imgseg = cv2.normalize(imgseg, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range((3)):
       img[:,:,z]=(img[:,:,z]*(imgseg))
    hista = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])
    imgseg=cv2.imread(segmenta,0)
    imgseg1=imgseg*-1
    imgseg1 = cv2.normalize(imgseg1, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    for z in range((3)):
       img2[:,:,z]=(img2[:,:,z]*(imgseg1))
    histb = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])
    #Euclidian Distance
    histograma = []
    histograma2 = []
    euclidian = []
    
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [1, 256])
        hist2 = cv2.calcHist([img2], [i], None, [256], [1, 256])
        histograma.append(hist)
        histograma2.append(hist2)

    for l in range(3):
        eucl = distance.euclidean(histograma[l], histograma2[l])
        euclidian.append(eucl)
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2    
    #Correlación
    correlacion=cv2.compareHist(hista,histb,cv2.HISTCMP_CORREL)
    #Dista euclidiana
    euclidiana = statistics.mean(euclidian)
    #Distancia Bhattacharyya
    Bhattacharyya=cv2.compareHist(hista,histb,cv2.HISTCMP_BHATTACHARYYA)
    
    #print('Correlación=', correlacion)
    #print('Distancia Bhattacharyya=', Bhattacharyya)
    return img,img2,hista,histb,correlacion,Bhattacharyya,euclidiana
