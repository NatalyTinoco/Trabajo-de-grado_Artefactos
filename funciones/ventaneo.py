# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:07:32 2019

@author: Nataly
"""
from matplotlib import pyplot as plt

def ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, V):
    vecesA = int(a/tamañoa1A)
    vecesB = int(b/tamañoa1B)
    for fa1 in range(0,a-tamañoa1A,tamañoa1A):
       for ca1 in range(0,b-tamañoa1B,tamañoa1B):       
            vecesA = int(a/tamañoa1A)
            vecesB = int(b/tamañoa1B)
            croppeda1 = V[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B]
            if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:
               if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
                     croppeda1= V[fa1:a,ca1:b]
               else:
                      croppeda1 = V[fa1:fa1+tamañoa1A,ca1:]
            if fa1+tamañoa1A==tamañoa1A*vecesA-tamañoa1A:
                 if ca1+tamañoa1B==tamañoa1B*vecesB-tamañoa1B:
                     croppeda1 = V[fa1:a,ca1:b]                     
                 else:
                     croppeda1 = V[fa1:,ca1:ca1+tamañoa1B] 
#    cropped = im[f:f+tamañoA,c:c+tamañoB]
#   
#    #test2[f:f+tamañoA,c:c+tamañoB]=test[f:f+tamañoA,c:c+tamañoB]
#    if c==tamañoB*vecesB-tamañoB:
#        cropped = im[f:f+tamañoA,c:]
#   
#        #test2[f:f+tamañoA,c:]=test[f:f+tamañoA,c:]
#    if f==tamañoA*vecesA-tamañoA:
#         #print('ola')
#         if c==tamañoB*vecesB-tamañoB:
#            cropped = im[f:,c:]
#   
#             #test2[f:,c:]=test[f:,c:]
#         else:
#             cropped = im[f:,c:c+tamañoB]
       
    return croppeda1