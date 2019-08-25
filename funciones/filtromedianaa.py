# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:25:38 2019

"""

def filtromediana(nuevaA,tama,tamanu,tipodemas,im):
    import numpy as np 
    import statistics as stats

    if tipodemas==3:
        fmediana=np.zeros((tama[0],tama[1]))
        ta=fmediana.shape
        ta=list(ta)
        for x in range (1,tamanu[0]-1) :
            for y in range(1,tamanu[1]-1):
                listd=[nuevaA[x-1,y-1],nuevaA[x-1,y],nuevaA[x-1,y+1],nuevaA[x,y-1],nuevaA[x,y],nuevaA[x,y+1],nuevaA[x+1,y-1],nuevaA[x+1,y],nuevaA[x+1,y+1]]
                #print(x,y)  
                listd.sort()
                im[x-1,y-1]=listd[4]
        fmediana=im
    if tipodemas==5:
        fmediana=np.zeros((tama[0],tama[1]))
        ta=fmediana.shape
        ta=list(ta)
        for x in range (2,tamanu[0]-2) :
            for y in range(2,tamanu[1]-2):
                listd=[nuevaA[x-2,y-2],nuevaA[x-2,y-1],nuevaA[x-2,y],nuevaA[x-2,y+1],nuevaA[x-2,y+2],nuevaA[x-1,y-2],nuevaA[x-1,y-1],nuevaA[x-1,y],nuevaA[x-1,y+1],nuevaA[x-1,y+2],nuevaA[x,y-2],nuevaA[x,y-1],nuevaA[x,y],nuevaA[x,y+1],nuevaA[x,y+2],nuevaA[x+1,y-2],nuevaA[x+1,y-1],nuevaA[x+1,y],nuevaA[x+1,y+1],nuevaA[x+1,y+2],nuevaA[x+2,y-2],nuevaA[x+2,y-1],nuevaA[x+2,y],nuevaA[x+2,y+1],nuevaA[x+2,y+2]]
                #print(x,y)
                #print(nuevaA[x,y-2])
                #print(listd)
                listd.sort()
                im[x-2,y-2]=listd[12]
        fmediana=im
    if tipodemas==11:
        fmediana=np.zeros((tama[0],tama[1]))
        ta=fmediana.shape
        ta=list(ta)
        for x in range (5,tamanu[0]-5) :
            for y in range(5,tamanu[1]-5):
                listd=[nuevaA[x-5,y-5],nuevaA[x-5,y-4],nuevaA[x-5,y-3],nuevaA[x-5,y-2],nuevaA[x-5,y-1],nuevaA[x-5,y],nuevaA[x-5,y+1],nuevaA[x-5,y+2],nuevaA[x-5,y+3],nuevaA[x-5,y+4],nuevaA[x-5,y+5],nuevaA[x-4,y-5],nuevaA[x-4,y-4],nuevaA[x-4,y-3],nuevaA[x-4,y-2],nuevaA[x-4,y-1],nuevaA[x-4,y],nuevaA[x-4,y+1],nuevaA[x-4,y+2],nuevaA[x-4,y+3],nuevaA[x-4,y+4],nuevaA[x-4,y+5],nuevaA[x-3,y-5],nuevaA[x-3,y-4],nuevaA[x-3,y-3],nuevaA[x-3,y-2],nuevaA[x-3,y-1],nuevaA[x-3,y],nuevaA[x-3,y+1],nuevaA[x-3,y+2],nuevaA[x-3,y+3],nuevaA[x-3,y+4],nuevaA[x-3,y+5],nuevaA[x-2,y-5],nuevaA[x-2,y-4],nuevaA[x-2,y-3],nuevaA[x-2,y-2],nuevaA[x-2,y-1],nuevaA[x-2,y],nuevaA[x-2,y+1],nuevaA[x-2,y+2],nuevaA[x-2,y+3],nuevaA[x-2,y+4],nuevaA[x-2,y+5],nuevaA[x-1,y-5],nuevaA[x-1,y-4],nuevaA[x-1,y-3],nuevaA[x-1,y-2],nuevaA[x-1,y-1],nuevaA[x-1,y],nuevaA[x-1,y+1],nuevaA[x-1,y+2],nuevaA[x-1,y+3],nuevaA[x-1,y+4],nuevaA[x-1,y+5],nuevaA[x,y-5],nuevaA[x,y-4],nuevaA[x,y-3],nuevaA[x,y-2],nuevaA[x,y-1],nuevaA[x,y],nuevaA[x,y+1],nuevaA[x,y+2],nuevaA[x,y+3],nuevaA[x,y+4],nuevaA[x,y+5],nuevaA[x+1,y-5],nuevaA[x+1,y-4],nuevaA[x+1,y-3],nuevaA[x+1,y-2],nuevaA[x+1,y-1],nuevaA[x+1,y],nuevaA[x+1,y+1],nuevaA[x+1,y+2],nuevaA[x+1,y+3],nuevaA[x+1,y+4],nuevaA[x+1,y+5],nuevaA[x+2,y-5],nuevaA[x+2,y-4],nuevaA[x+2,y-3],nuevaA[x+2,y-2],nuevaA[x+2,y-1],nuevaA[x+2,y],nuevaA[x+2,y+1],nuevaA[x+2,y+2],nuevaA[x+2,y+3],nuevaA[x+2,y+4],nuevaA[x+2,y+5],nuevaA[x+3,y-5],nuevaA[x+3,y-4],nuevaA[x+3,y-3],nuevaA[x+3,y-2],nuevaA[x+3,y-1],nuevaA[x+3,y],nuevaA[x+3,y+1],nuevaA[x+3,y+2],nuevaA[x+3,y+3],nuevaA[x+3,y+4],nuevaA[x+3,y+5],nuevaA[x+4,y-5],nuevaA[x+4,y-4],nuevaA[x+4,y-3],nuevaA[x+4,y-2],nuevaA[x+4,y-1],nuevaA[x+4,y],nuevaA[x+4,y+1],nuevaA[x+4,y+2],nuevaA[x+4,y+3],nuevaA[x+4,y+4],nuevaA[x+4,y+5],nuevaA[x+5,y-5],nuevaA[x+5,y-4],nuevaA[x+5,y-3],nuevaA[x+5,y-2],nuevaA[x+5,y-1],nuevaA[x+5,y],nuevaA[x+5,y+1],nuevaA[x+5,y+2],nuevaA[x+5,y+3],nuevaA[x+5,y+4],nuevaA[x+5,y+5]]
                #print(x,y)
                #print(nuevaA[x,y-2])
                #print(listd)
                listd.sort()
                fmediana[x-5,y-5]=listd[60]
        fmediana=im    
    return fmediana 