# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:07:32 2019

@author: Nataly
"""

def ventaneoo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, otra):
    vecesA = int(a/tamañoa1A)
    vecesB = int(b/tamañoa1B)
    croppeda1 = otra[fa1:fa1+tamañoa1A,ca1:ca1+tamañoa1B]
    if ca1==tamañoa1B*vecesB-tamañoa1B:
        croppeda1 = otra[fa1:fa1+tamañoa1A,ca1:b]
    if fa1==tamañoa1A*vecesA-tamañoa1A :
         if ca1==tamañoa1B*vecesB-tamañoa1B:
            croppeda1 = otra[fa1:,ca1:]
         else:
             croppeda1 = otra[fa1:,ca1:ca1+tamañoa1B]
       
    return croppeda1

def ventadibujo(tamañoa1A, tamañoa1B,a,b,fa1,ca1, original_3_1):
    import cv2
    vecesA = int(a/tamañoa1A)
    vecesB = int(b/tamañoa1B)
    
    if ca1==tamañoa1B*vecesB-tamañoa1B and ca1==tamañoa1B*vecesB-tamañoa1B or ca1==0 and fa1==0:
        cv2.rectangle(original_3_1,(int(ca1),int(fa1)),(int(ca1+tamañoa1B),int(fa1+tamañoa1A)),(255,0,0),2)  
#    else:and fa1!=tamañoa1A*vecesA-tamañoa1A
    if ca1==tamañoa1B*vecesB-tamañoa1B and fa1==0:
        cv2.rectangle(original_3_1,(int(ca1),int(fa1)),(int(ca1+tamañoa1B),int(b)),(255,0,0),2)  
    if fa1==tamañoa1A*vecesA-tamañoa1A and ca1==0:
         if ca1==tamañoa1B*vecesB-tamañoa1B:
            cv2.rectangle(original_3_1,(int(ca1),int(fa1)),(int(a),int(b)),(255,0,0),2) 
         else:
             cv2.rectangle(original_3_1,(int(ca1),int(fa1)),(int(ca1+tamañoa1A),int(b)),(255,0,0),2) 
   
    return original_3_1