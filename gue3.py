# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:18:25 2019

@author: Usuario
"""

import tkinter
import cv2
import PIL.Image, PIL.ImageTk
 
# Create a window
window = tkinter.Tk()
window.title("OpenCV and Tkinter")
 
# Load an image using OpenCV
cv_img = cv2.cvtColor(cv2.imread("C:/Users/Usuario/Documents/Daniela/IMG_2983.JPG"), cv2.COLOR_BGR2RGB)
 
# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, no_channels = cv_img.shape
 
# Create a canvas that can fit the above image
canvas = tkinter.Canvas(window, width = width, height = height)
canvas.pack()
 
# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
 
# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
 
# Run the window loop
window.mainloop()
