# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:13:39 2019

@author: Usuario
"""
import sys
sys.path.insert(1, './funciones')

from test_todoRE import test_all_RE
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image

HEIGHT = 700
WIDTH = 800    
<<<<<<< HEAD
=======

nameI = ''
>>>>>>> a8ba2bacc2fb7fe0aa760ab74202460018cfea3d

nameI = ''

class Txt(object):
    def SetValue(data): pass
    def GetValue(): pass
txt = Txt()

def identify():
    pred = test_all_RE(txt.GetValue())
#    print(pred)
    label_pred = tk.Label(pred_frame,text=str(pred))
    label_pred.place(relwidth=1,relheight=1)

def fileDialog():
    filename = filedialog.askopenfile(initialdir="/",title="Select a file",
                                      filetypes=(("jpg files","*.jpg"),("png files","*.png")))
    
    cv_img = cv2.cvtColor(cv2.imread(filename.name), cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img,(350,350),interpolation = cv2.INTER_AREA)
    photo = ImageTk.PhotoImage(image = Image.fromarray(cv_img))
    label = tk.Label(image_frame,image=photo)
    label.img = photo
    label.place(relwidth=1,relheight=1)
    label.pack()
    global nameI
    nameI = filename.name
    
    txt.SetValue(filename.name)
#%%
root = tk.Tk()

canvas = tk.Canvas(root,height=HEIGHT,width=WIDTH)
canvas.pack()

#frame = tk.Frame(root,bg='#80c1ff',bd=5)
frame = tk.Frame(root,bd=5)
frame.place(relx=0.46,rely=0.09,relwidth=0.9,relheight=0.1,anchor='n')

pred_frame = tk.Frame(root,bg='#80c1ff',bd=5)
pred_frame.place(relx=0.53,rely=0.35,relwidth=0.22,relheight=0.1,anchor='n')

image_frame = tk.Frame(root,bg='#42454d',bd=7)
image_frame.place(relx=0.16,rely=0.22,relwidth=0.3,relheight=0.4,anchor='n')

play_frame = tk.Frame(root,bd=5)
play_frame.place(relx=0.46,rely=0.65,relwidth=0.9,relheight=0.1,anchor='n')
    
#select_label = tk.Label(frame,text="Seleccionar una imagen",font=40)
#select_label.place(relx=0,rely=0.2,relheight=1,relwidth=0.3)

<<<<<<< HEAD
button = tk.Button(frame,text='Examinar',bg='#42454d',fg='#3eb3f9',font=50,command=fileDialog)
button.place(relx=0,relheight=1,relwidth=0.33)

button2 = tk.Button(play_frame,text='Analizar imagen',bg='#42454d',
                   fg='#3eb3f9',font=50,command=identify(nameI))
button2.place(relx=0,relheight=1,relwidth=0.33)
=======
def identify(fileName):
    pred = test_all_RE(fileName)
#    print(pred)
    label_pred = tk.Label(pred_frame,text=str(pred))
    label_pred.place(relwidth=1,relheight=1)

def fileDialog():
    filename = filedialog.askopenfile(initialdir="/",title="Select a file",
                                      filetypes=(("jpg files","*.jpg"),("png files","*.png")))
    
    cv_img = cv2.cvtColor(cv2.imread(filename.name), cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img,(350,350),interpolation = cv2.INTER_AREA)
    photo = ImageTk.PhotoImage(image = Image.fromarray(cv_img))
    label = tk.Label(image_frame,image=photo)
    label.img = photo
    label.place(relwidth=1,relheight=1)
    label.pack()
    global nameI
    nameI = filename.name

button = tk.Button(frame,text='Examinar',bg='#42454d',fg='#3eb3f9',font=50,command=fileDialog)
button.place(relx=0,relheight=1,relwidth=0.33)

if nameI != '':
    print(nameI)
#    button2 = tk.Button(play_frame,text='Analizar imagen',bg='#42454d',
#                       fg='#3eb3f9',font=50,command=identify(nameI))
#    button2.place(relx=0,relheight=1,relwidth=0.33)
>>>>>>> a8ba2bacc2fb7fe0aa760ab74202460018cfea3d

root.mainloop()