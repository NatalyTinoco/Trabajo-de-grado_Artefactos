# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:10:22 2019

"""
def yolo2voc(boxes, imshape):

    import numpy as np 
    m, n = imshape[:2]
    
    box_list = []
    for b in boxes:
        cls, x, y, w, h = b
        
        x1 = (x-w/2.)
        x2 = x1 + w
        y1 = (y-h/2.)
        y2 = y1 + h
        
        # absolute:
        x1 = x1 * n ; x2 = x2*n
        y1 = y1 * m ; y2 = y2*m
        
        box_list.append([cls, x1,y1,x2,y2])
    
    if len(box_list)>0:
        box_list = np.vstack(box_list)
        
    return box_list
#
