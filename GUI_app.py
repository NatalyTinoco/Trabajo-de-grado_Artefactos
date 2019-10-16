# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:19:33 2019

@author: Nataly
"""

import sys
sys.path.insert(1, './funciones')

from test_todoRE import test_all_RE
from test_todoDM import test_all_DM
from correccion import suavizado, inpaintingB, inpaintingNS, inpaintingTA

from PyQt5.QtGui import QIcon, QPixmap,QImage
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QFileDialog
from PIL import Image, ImageQt

from os import getcwd
import cv2
import numpy as np 


from PyQt5 import uic, QtWidgets

qtCreatorFile = "pruebaGUI_6.ui" #Aquí va el nombre de tu archivo

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
#
        # ================== EVENTOS QPUSHBUTTON ===================

        self.Examinar.clicked.connect(self.seleccionarImagen)
        self.Analizar.clicked.connect(self.analizaImagen)
        self.Corregir.clicked.connect(self.corregirImagen)
        self.ok.clicked.connect(self.guardarImagen)
#        
##        buttonEliminar.clicked.connect(lambda: self.labelImagen.clear())
#    
    def seleccionarImagen(self):
        global filePath,pixmapImagen
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos')
       
        if filePath != "":
            print ("Dirección",filePath) #Opcional imprimir la dirección del archivo
            if str(filePath):
            # Adaptar imagen
                pixmapImagen = QPixmap(str(filePath)).scaled(351, 291, Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation)
            # Mostrar imagen
            self.label_2.setPixmap(pixmapImagen)
#            self.show()
#            global imagen
#            imagen = cv2.imread(str(filePath))
    def analizaImagen(self):
        global filePath,original_2,imDU_2,umbrImage
        print ("======",filePath) #Opcional imprimir la dirección del archivo
        
        resul, original_2,imDU_2,umbrImage=test_all_RE(str(filePath))
        resuldm,originaldm_2,imDRdm_2=test_all_DM(str(filePath))
        if np.mean(resul)==1 and np.mean(resuldm)==1:
            self.textEdit.setText('Reflejos especulares')
        if np.mean(resul)==0 and np.mean(resuldm)==0:
            self.textEdit.setText('Desenfoque por movimiento')
        if np.mean(resul)==0 and np.mean(resuldm)==1:
            self.textEdit.setText('No tiene reflejos ni desenfoque')
        if np.mean(resul)!=0 and np.mean(resul)!=1 or np.mean(resuldm)!=0 and np.mean(resuldm)!=1 :
            self.textEdit.setText('Reflejos especulares y Desenfoque por movimiento')
            
        self.label_3.setPixmap(pixmapImagen)
        
    def corregirImagen(self):
        global filePath,original_2,imDU_2,umbrImage
        print ("======",filePath) #Opcional imprimir la dirección del archivo
        if self.automatica.isChecked():
            umbrImage=cv2.normalize(umbrImage, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            ipa=inpaintingNS(imDU_2,umbrImage)
            ipa=cv2.cvtColor(ipa, cv2.COLOR_RGB2BGR) 
            height, width, channel = ipa.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipa, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = QPixmap(qImg.scaled(351, 291, Qt.KeepAspectRatio,Qt.SmoothTransformation))
            
            self.label_3.setPixmap(qImg)
            
        if self.radioButton.isChecked():   
            nup=self.horizontalSlider.value()
            isu=suavizado(imDU_2,umbrImage,int(nup))
            ipa=cv2.cvtColor(isu, cv2.COLOR_RGB2BGR) 
            height, width, channel = ipa.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipa, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = QPixmap(qImg.scaled(351, 291, Qt.KeepAspectRatio,Qt.SmoothTransformation))
            self.label_3.setPixmap(qImg)
            
    def guardarImagen(self):
        item=self.cb.currentText()
        print(item)
                        
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    