# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:20:26 2019

@author: Nataly
"""
def read_img(imfile):
    import cv2
    return cv2.imread(imfile)
    #return (imfile + 2)
