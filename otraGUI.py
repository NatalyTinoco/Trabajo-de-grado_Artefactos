# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:13:39 2019

@author: Usuario
"""
import sys
sys.path.insert(1, './funciones')

from test_todoRE import testRE
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image

HEIGHT = 700
WIDTH = 800

root = tk.Tk()

canvas = tk.Canvas(root,height=HEIGHT,width=WIDTH)
canvas.pack()

#frame = tk.Frame(root,bg='#80c1ff',bd=5)
frame = tk.Frame(root,bd=5)
frame.place(relx=0.46,rely=0.09,relwidth=0.9,relheight=0.1,anchor='n')

pred_frame = tk.Frame(root,bg='#80c1ff',bd=5)
pred_frame.place(relx=0.53,rely=0.35,relwidth=0.22,relheight=0.1,anchor='n')

def fileDialog():
    filename = ''
    filename = filedialog.askopenfile(initialdir="/",title="Select a file",
                                      filetypes=(("jpg files","*.jpg"),("png files","*.png")))
    return filename.name

def getImage():  
    name = ''
    photo = []
    name= fileDialog()
    cv_img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img,(330,330),interpolation = cv2.INTER_AREA)
    photo = ImageTk.PhotoImage(image = Image.fromarray(cv_img))
    return photo,name

def idetify(fileName):
    pred = testRE(fileName)
    label_pred = tk.Label(pred_frame,text=str(pred))
    label_pred.place(relwidth=1,relheight=1)
    
#select_label = tk.Label(frame,text="Seleccionar una imagen",font=40)
#select_label.place(relx=0,rely=0.2,relheight=1,relwidth=0.3)

button = tk.Button(frame,text='Examinar',bg='#42454d',fg='#3eb3f9',font=50,command=fileDialog)
button.place(relx=0,relheight=1,relwidth=0.3)

image_frame = tk.Frame(root,bg='#42454d',bd=10)
image_frame.place(relx=0.16,rely=0.22,relwidth=0.3,relheight=0.4,anchor='n')

imagen, imageName = getImage()
label = tk.Label(image_frame,image=imagen)
label.place(relwidth=1,relheight=1)

play_frame = tk.Frame(root,bd=5)
play_frame.place(relx=0.46,rely=0.65,relwidth=0.9,relheight=0.1,anchor='n')

button = tk.Button(play_frame,text='Analizar imagen',bg='#42454d',fg='#3eb3f9',
                   font=50,command=idetify(imageName))
button.place(relx=0,relheight=1,relwidth=0.34)

root.mainloop()