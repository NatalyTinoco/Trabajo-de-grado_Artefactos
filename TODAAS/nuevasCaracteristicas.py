# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:50:24 2019

@author: Nataly
"""
import cv2
import numoy as np 

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def HOP(im):
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return 
def SingularValueFeature(A):
    from numpy import array
    from numpy import diag
    from numpy import zeros
    from scipy.linalg import svd
    # define a matrix
#    A = array([
#    	[1,2,3,4,5,6,7,8,9,10],
#    	[11,12,13,14,15,16,17,18,19,20],
#    	[21,22,23,24,25,26,27,28,29,30]])
#    print(A)
    # Singular-value decomposition
    U, s, VT = svd(A)
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[0], :A.shape[0]] = diag(s)
    # select
    n_elements = 2
    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    # reconstruct
    B = U.dot(Sigma.dot(VT))
#    print(B)
    # transform
    T = U.dot(Sigma)
#    print(T)
    T = A.dot(VT.T)
#    print(T)
    return  B,T

def estimate_blur(image, threshold=100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)


def pretty_blur_map(blur_map, sigma=5):
    abs_image = np.log(np.abs(blur_map).astype(np.float32))
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)

