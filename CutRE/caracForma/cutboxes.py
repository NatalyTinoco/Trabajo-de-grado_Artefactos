# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 00:13:41 2019

@author: Nataly
"""

def cut_boxes(boxes,im,r,d):
    #import pylab as plt 
    #from matplotlib import pyplot as plt
    import cv2
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 0 or cls == 3:
            print('ok')
            cropped = im[int(y1):int(y2),int(x1):int(x2)]
            #plt.imshow(cropped)
            #plt.show()
            #plt.imshow(im)
            #plt.plot([x2],[y2],'ro')
            #plt.axis([0,1349,1079,0])
            #plt.show()
            
            if cls==0:
                print('re')
                r=r+1
                dire="segme/bboxreflejo (%d).jpg" %(r)
            if cls==3:
                d=d+1
                dire="bboxdesenfoque (%d).jpg" %(d)
            #cv2.imwrite(imgfile+'Z.jpg', imgR)
            try:
                cv2.imwrite(dire,cropped)
            except:
                print('Failed saving image')
    return r,d