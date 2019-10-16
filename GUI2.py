# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 05:34:58 2019

@author: Usuario
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
 
HEIGHT = 400
WIDTH = 400

root = tk.Tk()

canvas = tk.Canvas(root,height=HEIGHT,width=WIDTH)
canvas.pack()

frame = tk.Frame(root)
#frame.place(relx=0.1,rely=0.1,relwidth=1,relheight=1)
frame.place(relwidth=1,relheight=1)

label = tk.Label(frame,text="Open a file")
#label.pack(side = 'top', pady = 10)
label.grid(column=0,row=1)

def fileDialog():
    filename = filedialog.askopenfile(initialdir="/",title="Select a file",
                                      filetypes=(("jpg files","*.jpg"),("png files","*.png")))
    
#    label = ttk.Label(frame,text="")
#    label.grid(column=0,row=3)
#    label.configure(text=filename.name)
    print(filename.name)
    return filename.name
    
def getImage():  
    name= fileDialog()
    cv_img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img,(WIDTH,HEIGHT),interpolation = cv2.INTER_AREA)
    photo = ImageTk.PhotoImage(image = Image.fromarray(cv_img))
    return photo

imagen = getImage()
button = tk.Button(frame,text='Browse a File', command=fileDialog)
button.grid(column=0,row=2)

my_label = tk.Label(frame,image=imagen,height=HEIGHT,width=WIDTH)
my_label.grid(column=0,row=4,padx=20, pady=20)


root.mainloop()
