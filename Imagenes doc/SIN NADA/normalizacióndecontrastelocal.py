# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import cv2
import matplotlib.pyplot as plt
def normalizacionlocalcontraste(image):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import PIL.Image as pim
    #matplotlib inline
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    #image = plt.imread('00025.jpg')
    #plt.imshow(image)
    #plt.show()
    image_tensor = torch.Tensor([np.array(image).transpose((2,0,1))])
    image_tensor.shape
    radius = 3
    #Gaussian Kernel
    def gaussian_filter(kernel_shape):
        x = np.zeros(kernel_shape, dtype='float32')
     
        def gauss(x, y, sigma=2.0):
            Z = 2 * np.pi * sigma ** 2
            return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
     
        mid = np.floor(kernel_shape[-1] / 2.)
        for kernel_idx in range(0, kernel_shape[1]):
            for i in range(0, kernel_shape[2]):
                for j in range(0, kernel_shape[3]):
                    x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)
     
        return x / np.sum(x)
    gfilter = torch.Tensor(gaussian_filter((1,3,9,9)))
    plt.contour(gfilter.numpy()[0][2])
    #Applying the Convolution
    filtered = F.conv2d(image_tensor,gfilter,padding=8) ## padding = 8 = 9-1 (radius - 1 ) for border == 'full'
    filtered.byte().numpy()[0][0].shape
    #The filtered output
    #plt.imshow(filtered[0][0],cmap='gray')
    #Removing the padding boundary added
    mid = int(np.floor(gfilter.shape[2] / 2.))
    filtered[:,:,mid:-mid,mid:-mid].shape
    #plt.imshow(filtered[:,:,mid:-mid,mid:-mid][0][0],cmap='gray')
    #Centered Image
    centered_image = image_tensor - filtered[:,:,mid:-mid,mid:-mid]
    centered_image[0].mean()  ##  mean is close 0 (If we take a finer filter, the mean will tend to 0)
    centered_image.shape
    centered_image[0].numpy().shape
    pf = centered_image[0].numpy().transpose((1,2,0))
    #plt.imshow(pf)
    #plt.show()
    imanorm=((pf - pf.min())/(pf.max() - pf.min()))
    imanorm=imanorm
    #plt.imshow(imanorm)   ## Scaled between 0 to 1
    #plt.show()    
    #print(pf.min(),pf.max())
    #print(imanorm.min(),imanorm.max())
    return imanorm

"""
file='Lap_01004'
file='WL_00485'
#file='00000'
#file='norm (1)'
#file='NCL (532)'
image = plt.imread(file+'.jpg')
plt.imshow(image)
plt.show()
print('Min: %.3f, Max: %.3f' % (image.min(), image.max()))
plt.hist(image.ravel(),256,[image.min(),image.max()]); plt.show()

imanorm=normalizacionlocalcontraste(image)
imanorm=cv2.cvtColor(imanorm,cv2.COLOR_BGR2RGB)
plt.imshow(imanorm)
plt.show()
print('Min: %.3f, Max: %.3f' % (imanorm.min(), imanorm.max()))
plt.hist(imanorm.ravel(),256,[imanorm.min(),imanorm.max()]); plt.show()

#cv2.imshow('',imanorm)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 

#show()
"""



