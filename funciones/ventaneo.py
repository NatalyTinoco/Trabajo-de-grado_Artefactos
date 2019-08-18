# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:07:32 2019

@author: Nataly
"""
from matplotlib import pyplot as plt

def ventaneoo(tamañoA, tamañoB,a,b,f,c, im):
    vecesA = int(a/tamañoA)
    vecesB = int(b/tamañoB)
    cropped = im[f:f+tamañoA,c:c+tamañoB]
   
    #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
    if c==tamañoB*vecesB-tamañoB:
        cropped = im[f:f+tamañoA,c:]
   
        #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
    if f==tamañoA*vecesA-tamañoA:
         #print('ola')
         if c==tamañoB*vecesB-tamañoB:
            cropped = im[f:,c:]
   
             #test2[f:,c:]=test[f:,c:]
         else:
             cropped = im[f:,c:c+tamañoB]
             #test2[f:,c:c+tamañoB]=test[f:,c:c+tamañoB]
             #print('dani')
#            plt.imshow(cropped)
#            plt.show() 
#            print('h',h)
    return cropped