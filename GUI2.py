# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 05:34:58 2019

@author: Usuario
"""

import tkinter as tk
from tkinter import filedialog

HEIGHT = 700
WIDTH = 800

root = tk.Tk()

canvas = tk.Canvas(root,height=HEIGHT,width=WIDTH)
canvas.pack()

frame = tk.Frame(root)
#frame.place(relx=0.1,rely=0.1,relwidth=1,relheight=1)
frame.place(relwidth=1,relheight=1)

label = tk.Label(frame,text="Open a file")
label.grid(column=0,row=1)

def fileDialog():
    filedialog.askopenfile(initialdir="/",title="Select a file",
                                      filetypes=(("png files","*.png"), ("jpg files","*.jpg")))
    
    
button = tk.Button(frame,text='Browse a File', command=fileDialog)
button.grid(column=0,row=2)

root.mainloop()
