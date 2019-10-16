# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:00:13 2019

@author: Usuario
"""
#import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

class Root(tkinter.Tk):
    def _init_(self):
        super(Root, self)._init_()
        self.title("Detección de RE y DM")
        self.minsize(600,400)
        self.wm_iconbitmap('icon.ico')
        
        self.labelFrame = ttk.Labelframe(self,text='Open a File')
        self.labelFrame.grid(column=0,row=1,padx=20,pady=20)
        
        self.button()
        
        
    def button(self):
        self.button=ttk.Button(self.labelFrame, text='Browse a File')
        self.button.grid(column=1,row=1)

if __name__ == '__main__':
    root=Root()
    root.mainloop()
