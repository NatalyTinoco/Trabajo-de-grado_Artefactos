# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:44:13 2019

@author: Nataly
"""
def compararDM(img,boxes,imgfile,imgC,a):
    import numpy as np 
    import cv2
    from yolovoc import yolo2voc #conversion format ## box_list=yolo2voc(boxes, imshape) ##
    boxes_abs = yolo2voc(boxes, img.shape)  
    ta=img.shape
    ta=list(ta)
    mascara=np.zeros([ta[0],ta[1]])
    mascara2=np.ones([ta[0],ta[1]])
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            if a==0:
                artefacto=img[int(y1):int(y2),int(x1):int(x2),:]
            else:
                artefacto=img[int(y1):int(y2),int(x1):int(x2)]
            tavta=artefacto.shape
            tavta=list(tavta)              
            for y in range(int(y1),int(y2)):
                for x in range(int(x1),int(x2)):
                    mascara[y,x]=1
                    mascara2[y,x]=0
            mascara=mascara.astype(np.uint8)
            mascara2=mascara2.astype(np.uint8)
    img1=img
    #print(ta)
    #print(imgC.shape)
    if a==0:
        for z in range((3)):
            img1[:,:,z]=(img1[:,:,z]*(mascara))  
        img2=imgC
        for z in range((3)):
            img2[:,:,z]=(img2[:,:,z]*(mascara2))  
        hista = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])
        histb = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[1, 256, 1, 256, 1, 256])   
           
    else:
        #print(img1.shape,mascara.shape)
        img1=img1*mascara
        img2=imgC*mascara2
        hista = cv2.calcHist([img1], [0], None, [256], [1, 256])
        histb = cv2.calcHist([img2], [0], None, [256], [1, 256])   
    """if a==2:
        img1=img1*mascara
        img2=imgC[:,:,1]*mascara2
        hista = cv2.calcHist([img1], [0], None, [256], [0, 256])
        histb = cv2.calcHist([img2], [0], None, [256], [0, 256])   
    if a==3:
        img1=img1*mascara
        img2=imgC[:,:,2]*mascara2
        hista = cv2.calcHist([img1], [0], None, [256], [0, 256])
        histb = cv2.calcHist([img2], [0], None, [256], [0, 256])   
       
    """
    
    return img1,img2,hista,histb