# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:26:25 2019

"""
def agregarceros(imD,tipomas):
    import numpy as np 
    tama=imD.shape
    tama=list(tama)
    hola=tipomas-1
    #print(tama)
    nueva=np.zeros((tama[0]+hola,tama[1]+hola))
    #print(nueva.shape)
    tamanu=nueva.shape
    tamanu=list(tamanu)
    for x in range (tama[0]) :
        for y in range(tama[1]):
            if tipomas==3:
                nueva[x+1,y+1]=imD[x,y]
            if tipomas==5:
                nueva[x+2,y+2]=imD[x,y]
            if tipomas==11:
                nueva[x+5,y+5]=imD[x,y]
    return nueva,tama,tamanu
