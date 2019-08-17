# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:50:24 2019

@author: Nataly
"""
import cv2
import numpy as np 

#def variance_of_laplacian(image):
#	 varla=cv2.Laplacian(image, cv2.CV_64F).var()
#    return varla

def HOP(image):
#    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
#    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
#    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    import matplotlib.pyplot as plt
    from skimage.feature import hog
    from skimage import data, exposure
    #image = data.astronaut()
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
#    ax1.axis('off')
#    ax1.imshow(image, cmap=plt.cm.gray)
#    ax1.set_title('Input image')
#    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
#    ax2.axis('off')
#    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#    ax2.set_title('Histogram of Oriented Gradients')
#    plt.show()
    return hog_image_rescaled

def SingularValueFeature(A):
    #import numpy.linalg as svd 
    k,k1=A.shape
    U,s,V=np.linalg.svd(A,full_matrices=False)
    #print(U.shape,s.shape,V.shape)
    reconst_matrix=np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return  reconst_matrix,s

def estimate_blur(image, threshold=100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score


def pretty_blur_map(blur_map, sigma=5):
    abs_image = np.log(np.abs(blur_map).astype(np.float32))
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)



#import glob
##import cv2
#i=0
import glob
from yolovoc import yolo2voc
from readboxes import read_boxes
from matplotlib import pyplot as plt
from rOI import ROI
from scipy import stats
from scipy import integrate

def Fourier(inA):
    f = np.fft.fft2(inA)
    fshift = np.fft.fftshift(f)
    fourier = 20*np.log(np.abs(fshift))
    fourier=fourier.astype(np.uint8)
    return fourier 
from scipy.signal import find_peaks
for image in glob.glob("*.jpg"):   
    im = cv2.imread(image)
    im=cv2.normalize(im, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    aa,bb,c = im.shape    
    imaROI=ROI(im)
    imaROI=cv2.normalize(imaROI, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    HSV=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(HSV)
    V=V*imaROI
        
    for z in range(c):
        im[:,:,z]=im[:,:,z]*imaROI
    
    
    _,contours,_= cv2.findContours(imaROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x3,y3,w3,h3 = cv2.boundingRect(cnt)
    filetxt=image[0:len(image)-3]+'txt'      
    bboxfile=filetxt
    boxes = read_boxes(bboxfile)
    boxes_abs = yolo2voc(boxes, im.shape)  
    re=0
    dm=0
    imunda=0
    imSinBBOX=im.copy()
    tamañoA = 500
    tamañoB = 500
    for b in boxes_abs:
        cls, x1, y1, x2, y2 = b
        if cls == 3:
            print('========DM============')
            dm=dm+1   
            a,b= V[int(y1):int(y2),int(x1):int(x2)].shape
            V1= V[int(y1):int(y2),int(x1):int(x2)]
            vecesA = int(a/tamañoA)
            vecesB = int(b/tamañoB)        
            for f in range(0,a-tamañoA,tamañoA):
                for c in range(0,b-tamañoB,tamañoB):
                    cropped = V1[f:f+tamañoA,c:c+tamañoB]
                    croppedrgb = im[f:f+tamañoA,c:c+tamañoB]
                    if c==tamañoB*vecesB-tamañoB:
                        cropped = V1[f:f+tamañoA,c:]
                        croppedrgb = im[f:f+tamañoA,c:]
                    if f==tamañoA*vecesA-tamañoA:
                         if c==tamañoB*vecesB-tamañoB:
                            cropped = V1[f:,c:]
                            croppedrgb = im[f:,c:]
                         else:
                             cropped = V1[f:,c:c+tamañoB]
                             croppedrgb = im[f:,c:c+tamañoB]     
                    cropped=cv2.resize(cropped,(500,500))
                    croppedrgb=cv2.resize(croppedrgb,(500,500))
                    cropped=Fourier(cropped)
                    hist = cv2.calcHist([cropped],[0],None,[256],[0,255])
                    plt.plot(hist)
                    plt.show()
                    hisa=hist.copy()
                    hist=hist.tolist() 
                    u=np.max(hist)
                    hi=hist.index(u)
                    from scipy.integrate import simps
                    from numpy import trapz
                    
                    
                    # The y values.  A numpy array is used here,
                    # but a python list could also be used.
                    y = np.array(hist)
                    
                    # Compute the area using the composite trapezoidal rule.
                    area = trapz(y, dx=5)
                    print("area =", sum(area))
                    
                    # Compute the area using the composite Simpson's rule.
                    area = simps(y, dx=5)
                    print("area =", sum(area))
                    
                    
#                    blur_map, score=estimate_blur(croppedrgb, threshold=100)
#                    plt.imshow(blur_map,'Greys')
#                    plt.show()
#                    sobelx = cv2.Sobel(cropped,cv2.CV_64F,1,0,ksize=5)
#                    sobely = cv2.Sobel(cropped,cv2.CV_64F,0,1,ksize=5)
#                    plt.plot(sobelx,sobely,'*')
#                    plt.show()
        if cls==0:
            re=re+1        
            print(re)
        if cls==2:
            imunda=imunda+1

    if re > 0 and dm==0:
        print('================RE==================')
        inta=V[y3:y3+h3,x3:x3+w3]
        im2=im[y3:y3+h3,x3:x3+w3]
        a,b=inta.shape       
        vecesA = int(a/tamañoA)
        vecesB = int(b/tamañoB)
        
        for f in range(0,a-tamañoA,tamañoA):
            for c in range(0,b-tamañoB,tamañoB):
                cropped2 = inta[f:f+tamañoA,c:c+tamañoB]
                croppedrgb2 = im2[f:f+tamañoA,c:c+tamañoB]
                if c==tamañoB*vecesB-tamañoB:
                    cropped2 = inta[f:f+tamañoA,c:]
                    croppedrgb2 = im2[f:f+tamañoA,c:]
                if f==tamañoA*vecesA-tamañoA:
                     if c==tamañoB*vecesB-tamañoB:
                        cropped2 = inta[f:,c:]
                        croppedrgb2 = im2[f:,c:]
                     else:
                         cropped2 = inta[f:,c:c+tamañoB]
                         croppedrgb2 = im2[f:,c:c+tamañoB]
                cropped2=cv2.resize(cropped2,(500,500))
                croppedrgb2=cv2.resize(croppedrgb2,(500,500))  
#                blur_map, score=estimate_blur(croppedrgb2, threshold=100)
#                plt.imshow(blur_map,'Greys')
#                plt.show()
                cropped2=Fourier(cropped2)
                hist = cv2.calcHist([cropped2],[0],None,[256],[0,255])
                plt.plot(hist)
                plt.show()
  
#                sobelx = cv2.Sobel(cropped2,cv2.CV_64F,1,0,ksize=5)
#                sobely = cv2.Sobel(cropped2,cv2.CV_64F,0,1,ksize=5)
#                plt.plot(sobelx,sobely)
#                plt.show()
                

                
    if re==0 and dm==0 and imunda==0:
        print('==============NO=============0')
        inta2=V[y3:y3+h3,x3:x3+w3]
        im3=im[y3:y3+h3,x3:x3+w3]
        a,b=inta2.shape
        vecesA = int(a/tamañoA)
        vecesB = int(b/tamañoB)
        
        for f in range(0,a-tamañoA,tamañoA):
            for c in range(0,b-tamañoB,tamañoB):
                cropped3 = inta2[f:f+tamañoA,c:c+tamañoB]
                croppedrgb3 = im3[f:f+tamañoA,c:c+tamañoB]
                if c==tamañoB*vecesB-tamañoB:
                    cropped3 = inta2[f:f+tamañoA,c:]
                    croppedrgb3 = im3[f:f+tamañoA,c:]
                if f==tamañoA*vecesA-tamañoA:
                     if c==tamañoB*vecesB-tamañoB:
                        cropped3 = inta2[f:,c:]
                        croppedrgb3 = im3[f:,c:]
                     else:
                         cropped3 = inta2[f:,c:c+tamañoB]
                         croppedrgb3 = im3[f:,c:c+tamañoB]
                cropped3=cv2.resize(cropped3,(500,500))
                croppedrgb3=cv2.resize(croppedrgb3,(500,500))
                cropped3=Fourier(cropped3)
                hist = cv2.calcHist([cropped3],[0],None,[256],[0,255])
                plt.plot(hist)
                plt.show()
                from scipy.integrate import simps
                from numpy import trapz
                
                
                # The y values.  A numpy array is used here,
                # but a python list could also be used.
                y = np.array(hist)
                
                # Compute the area using the composite trapezoidal rule.
                area = trapz(y, dx=5)
                print("area =", sum(area))
                
                # Compute the area using the composite Simpson's rule.
                area = simps(y, dx=5)
                print("area =", sum(area))
#                blur_map, score=estimate_blur(croppedrgb3, threshold=100)
#                plt.imshow(blur_map,'Greys')
#                plt.show()
#                sobelx = cv2.Sobel(cropped3,cv2.CV_64F,1,0,ksize=5)
#                sobely = cv2.Sobel(cropped3,cv2.CV_64F,0,1,ksize=5)
#                plt.plot(sobelx,sobely)
#                plt.show()

