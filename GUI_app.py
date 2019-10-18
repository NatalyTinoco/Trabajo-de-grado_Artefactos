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
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QFileDialog, QMessageBox, QAction
from PIL import Image, ImageQt

from os import getcwd
import cv2
import numpy as np 

global contador
contador=0
from PyQt5 import uic, QtWidgets

qtCreatorFile = "Artefactos.ui" #Aquí va el nombre de tu archivo

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
        self.pushButton.clicked.connect(self.guardarImagen)
#        
        self.Corregir.hide()
        self.label.hide()
        self.automatica.hide()
        self.radioButton.hide()
        self.horizontalSlider.hide()
        self.popup.hide()
        self.popup_DM.hide()
        self.popup_NO.hide()
        self.popup_DOS.hide()
        self.pushButton.hide()
        

        self.resize(400, 490)
        self.move(90,50)
##        buttonEliminar.clicked.connect(lambda: self.labelImagen.clear())
#    
    def seleccionarImagen(self):
        global filePath,pixmapImagen,contador
        contador=0
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
#        print ("======",filePath) #Opcional imprimir la dirección del archivo
       
        resul, original_2,imDU_2,umbrImage=test_all_RE(str(filePath))
#        print('=re=',resul)
        resuldm,originaldm_2,imDRdm_2=test_all_DM(str(filePath))
#        print('=dm=',resuldm)
        
        if np.mean(resul)==1 and np.mean(resuldm)==1:
            self.textEdit.setText('Reflejos especulares')
            self.Corregir.show()
            self.label.show()
            self.automatica.show()
            self.radioButton.show()
            self.horizontalSlider.show()
            self.popup.show()
            self.popup_DM.hide()
            self.pushButton.show()
            
        if np.mean(resul)==0 and np.mean(resuldm)==0:
            self.textEdit.setText('Desenfoque por movimiento')
            self.Corregir.hide()
            self.label.hide()
            self.automatica.hide()
            self.radioButton.hide()
            self.horizontalSlider.hide()
            self.popup.hide()
            self.pushButton.show()
            self.popup_DM.show()
            self.popup_NO.hide()
            self.popup_DOS.hide()
            
        if np.mean(resul)==0 and np.mean(resuldm)==1:
            self.textEdit.setText('No tiene reflejos ni desenfoque')
            self.Corregir.hide()
            self.label.hide()
            self.automatica.hide()
            self.radioButton.hide()
            self.horizontalSlider.hide()
            self.popup.hide()
            self.pushButton.show()
            self.popup.hide()
            self.popup_DM.hide()
            self.popup_NO.show()
            self.popup_DOS.hide()
            
        if np.mean(resul)!=0 and np.mean(resul)!=1 or np.mean(resuldm)!=0 and np.mean(resuldm)!=1 :
            self.textEdit.setText('Reflejos especulares y Desenfoque por movimiento')
            self.Corregir.show()
            self.label.show()
            self.automatica.show()
            self.radioButton.show()
            self.horizontalSlider.show()
            self.pushButton.show()
            self.popup.hide()
            self.popup_DM.hide()
            self.popup_NO.hide()
            self.popup_DOS.show()
            
        self.label_3.setPixmap(pixmapImagen)
        self.resize(1207, 490)
        
    def closeEvent(self,event):
      reply =QMessageBox.question(self, "Mensaje", "Seguro quiere salir", QMessageBox.Yes, QMessageBox.No)
       
      if reply ==QMessageBox.Yes:
       event.accept()
      else: 
       event.ignore()     
       
    def corregirImagen(self):
        global filePath,original_2,imDU_2,umbrImage, contador,ipa
        contador+=1
        should_delete = QMessageBox.question(self, "Corregir", "¿Desea corregir la imagen original (Yes) o la imagen sin etiquetas (No) ?", QMessageBox.Yes, QMessageBox.No)
        if should_delete == QMessageBox.Yes:
            imcoor=original_2
        else:
           imcoor=imDU_2
           
        if self.automatica.isChecked():
            umbrImage=cv2.normalize(umbrImage, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            ipa=inpaintingNS(imcoor,umbrImage)
            ipa=cv2.cvtColor(ipa, cv2.COLOR_RGB2BGR) 
            height, width, channel = ipa.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipa, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = QPixmap(qImg.scaled(351, 291, Qt.KeepAspectRatio,Qt.SmoothTransformation))
            
            self.label_3.setPixmap(qImg)
            
        if self.radioButton.isChecked():   
            nup=self.horizontalSlider.value()
            isu=suavizado(imcoor,umbrImage,int(nup))
            ipa=cv2.cvtColor(isu, cv2.COLOR_RGB2BGR) 
            height, width, channel = ipa.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipa, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = QPixmap(qImg.scaled(351, 291, Qt.KeepAspectRatio,Qt.SmoothTransformation))
            self.label_3.setPixmap(qImg)
            
    def guardarImagen(self):
        item=self.popup.currentText()
        item_dm=self.popup_DM.currentText()
        item_no=self.popup_NO.currentText()
        item_dos=self.popup_DOS.currentText()

        
        print(item)
        global name
        
        if item=='Guardar imagen original' or  item_dm=='Guardar imagen original'or  item_no=='Guardar imagen original' or  item_dos=='Guardar imagen original':
            formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
            name = QFileDialog.getSaveFileName(self, "Save as image", "untitled.jpg", formats)
            print('NAME',name)
            pixmapImagen_2 = QPixmap(str(filePath))
            pixmapImagen_2.save(str(name[0]))

        if item=='Guardar imagen corregida'  or  item_dos=='Guardar imagen corregida':
            if contador!=0:
                formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
                name = QFileDialog.getSaveFileName(self, "Save as image", "untitled.jpg", formats)
                print('NAME',name)
                height, width, channel = ipa.shape
                bytesPerLine = 3 * width
                qImg = QImage(ipa, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmapImagen_2= QPixmap(qImg)
                pixmapImagen_2.save(str(name[0]))
            else:
                QMessageBox.information(self,"pushButton", "No ha corregido la imagen.", QMessageBox.Ok)
                
        if item=='Descargar mascara binaria de RE'or  item_dos=='Descargar mascara binaria de RE':
                formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
                name = QFileDialog.getSaveFileName(self, "Save as image", "untitled.jpg", formats)
                global umbrImage
                umbrImage_2 = cv2.normalize(umbrImage, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                upa=umbrImage_2
                cv2.imwrite(str(name[0]),upa)
                

        if item=='Guardar imagen sin etiquetas' or item_dm=='Guardar imagen sin etiquetas' or  item_no=='Guardar imagen sin etiquetas' or  item_dos=='Guardar imagen sin etiquetas':
                formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
                name = QFileDialog.getSaveFileName(self, "Save as image", "untitled.jpg", formats)
                upa1=imDU_2
                upa1=cv2.cvtColor(upa1, cv2.COLOR_RGB2BGR) 
                height, width,cha = upa1.shape
                bytesPerLine = 3 * width
                qImg = QImage(upa1, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmapImagen_2= QPixmap(qImg)
                pixmapImagen_2.save(str(name[0]))         
                
        if item_dm=='Eliminar imagen' or  item_dos=='Eliminar imagen':   
            should_delete = QMessageBox.question(self, "pushButton", "¿Ésta seguro que desea eliminar este archivo?", QMessageBox.Yes, QMessageBox.No)
            print(should_delete)
            if should_delete == QMessageBox.Yes:
               from os import remove
               remove(str(filePath))
           
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    