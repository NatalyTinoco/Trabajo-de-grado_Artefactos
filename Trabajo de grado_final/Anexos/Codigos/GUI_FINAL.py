# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:19:33 2019

@author:Daniela y Nataly

Descripción del código: Código de la interfaz gráfica de usuario, donde se une todo el método de tratamiento de artefactos propuesto en este trabajo de grado.
"""
import sys
sys.path.insert(1, './codigos')
import math
from test_todoRE import test_all_RE
from test_todoDM import test_all_DM
from correccion import inpaintingTA
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import numpy as np 
from PyQt5 import uic, QtWidgets

global contador,otroc
otroc=0
contador=0
qtCreatorFile = "Artefactos_Final.ui" 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.Examinar.clicked.connect(self.seleccionarImagen)
        self.Analizar.clicked.connect(self.analizaImagen)
        self.pushButton.clicked.connect(self.guardarImagen)
        self.ubicacion.clicked.connect(self.mostrariubi)   
        self.corre.clicked.connect(self.mostraricorre) 
        self.popup.hide()
        self.popup_DM.hide()
        self.popup_NO.hide()
        self.popup_DOS.hide()
        self.pushButton.hide()
        self.RE.hide()
        self.DM.hide()
        self.bRE.hide()
        self.bDM.hide()
        self.ubicacion.hide()
        self.corre.hide()
        self.opciones.hide()
        self.resize(400, 490)
        self.move(90,50)

    def seleccionarImagen(self):
        global filePath,pixmapImagen,contador,w,h 
        w=350
        h=320
        contador=0
        formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', './Anexos/Codigos/',formats)
        if filePath != "":
            if str(filePath):
                pixmapImagen = QPixmap(str(filePath)).scaled(w, h, Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation)
            self.label_2.setPixmap(pixmapImagen)
       
    def analizaImagen(self):
        global filePath,original_2,imDU_2,umbrImage,otroc,contador,ipaOri,ipaetique,ipaubi     
        otroc+=1
        resul, original_2,imDU_2,umbrImage,original_3,bboxre=test_all_RE(str(filePath))
        resuldm,originaldm_2,imDRdm_2,original_3,bboxdm=test_all_DM(str(filePath),original_3)
        grupo0 = [i for i,x in enumerate(resul) if x == 0]
        grupo1 = [i for i,x in enumerate(resul) if x == 1]
        grupodm0 = [i for i,x in enumerate(resuldm) if x == 0]
        grupodm1 = [i for i,x in enumerate(resuldm) if x == 1]
        if len(grupo1)>0 and len(grupodm0)==0:
            self.RE.setText('Reflejos especulares')
            self.popup.show()
            self.popup_DM.hide()
            self.popup_DOS.hide()
            self.popup_NO.hide()
            self.pushButton.show()
            self.RE.show()
            self.DM.hide()
            self.bRE.show()
            self.bDM.hide()
            self.ubicacion.show()
            self.corre.show()
            self.opciones.show()
            fileRE='./botonRE.png'
            pixmapImagenRE = QPixmap(str(fileRE)).scaled(31, 31, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.bRE.setPixmap(pixmapImagenRE)
            
        if len(grupodm0)>0 and len(grupo1)==0:
            self.RE.setText('Desenfoque por movimiento')
            self.popup.hide()
            self.pushButton.show()
            self.popup_DM.show()
            self.popup_NO.hide()
            self.popup_DOS.hide()
            self.DM.hide()
            self.RE.show()
            self.bDM.hide()
            self.bRE.show()
            self.ubicacion.show()
            self.corre.hide()
            self.opciones.show()
            fileDM='./botonDM.png'
            pixmapImagenRE = QPixmap(str(fileDM)).scaled(31, 31, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.bRE.setPixmap(pixmapImagenRE)
            
        if  len(grupodm0)>0 and len(grupo1)>0 :
            self.RE.setText('Reflejos especulares')
            self.DM.setText('Desenfoque por movimiento')
            self.popup.hide()
            self.popup_DM.hide()
            self.popup_NO.hide()
            self.popup_DOS.show()
            self.pushButton.show()
            self.RE.show()
            self.DM.show()
            self.bRE.show()
            self.bDM.show()
            self.ubicacion.show()
            self.corre.show()
            self.opciones.show()
            fileRE='./botonRE.png'
            pixmapImagenRE = QPixmap(str(fileRE)).scaled(31, 31, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.bRE.setPixmap(pixmapImagenRE)
            fileDM='./botonDM.png'
            pixmapImagenDM = QPixmap(str(fileDM)).scaled(31, 31, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.bDM.setPixmap(pixmapImagenDM)
            
        if len(grupo0)>0 and len(grupodm1)>0 and len(grupo1)==0 and len(grupodm0)==0 or math.isnan(np.mean(resul))==True and len(grupodm1)>0 and len(grupodm0)==0:
            self.RE.setText('No tiene reflejos ni desenfoque')
            self.popup.hide()
            self.pushButton.show()
            self.popup.hide()
            self.popup_DM.hide()
            self.popup_NO.show()
            self.popup_DOS.hide()
            self.RE.show()
            self.DM.hide()
            self.bRE.hide()
            self.bDM.hide()
            self.ubicacion.hide()
            self.corre.hide()
            self.opciones.show()          
       
        umbrImage=cv2.normalize(umbrImage, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        ipaOri=inpaintingTA(original_2,umbrImage)
        umbrImage=cv2.normalize(umbrImage, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        ipaetique=inpaintingTA(imDU_2,umbrImage)
        ipaOri=cv2.cvtColor(ipaOri, cv2.COLOR_RGB2BGR) 
        ipaetique=cv2.cvtColor(ipaetique, cv2.COLOR_RGB2BGR) 
        ipaubi=cv2.cvtColor(original_3, cv2.COLOR_RGB2BGR) 
        height, width, channel = ipaubi.shape
        bytesPerLine = 3 * width
        qImg = QImage(ipaubi, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = QPixmap(qImg.scaled(w,h, Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.label_3.setPixmap(qImg)
        self.ubicacion.setStyleSheet('QPushButton { background-color: rgb(102, 102, 102);}')
        self.corre.setStyleSheet('QPushButton {background-color: rgb(175, 175, 175);}')
        self.resize(1307, 490)
        
    def mostrariubi(self):
        global ipaubi
        height, width, channel = ipaubi.shape
        bytesPerLine = 3 * width
        qImg = QImage(ipaubi, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = QPixmap(qImg.scaled(w,h, Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.label_3.setPixmap(qImg)
        self.ubicacion.setStyleSheet('QPushButton { background-color: rgb(102, 102, 102);}')
        self.corre.setStyleSheet('QPushButton {background-color: rgb(175, 175, 175);}')
        
    def mostraricorre(self):
        global ipaOri,ipaetique       
        height, width, channel = ipaOri.shape
        bytesPerLine = 3 * width
        qImg = QImage(ipaOri, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = QPixmap(qImg.scaled( w, h, Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.label_3.setPixmap(qImg)
        self.corre.setStyleSheet('QPushButton { background-color: rgb(102, 102, 102);}')
        self.ubicacion.setStyleSheet('QPushButton {background-color: rgb(175, 175, 175);}')  
        item=self.popup.currentText()
        item_dos=self.popup_DOS.currentText()
        
        if item=='Guardar imagen original corregida'  or  item_dos=='Guardar imagen original corregida':
            height, width, channel = ipaOri.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipaOri, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = QPixmap(qImg.scaled( w,h, Qt.KeepAspectRatio,Qt.SmoothTransformation))
            self.label_3.setPixmap(qImg)
            self.corre.setStyleSheet('QPushButton { background-color: rgb(102, 102, 102);}')
            self.ubicacion.setStyleSheet('QPushButton {background-color: rgb(175, 175, 175);}')
            
        if item=='Guardar imagen sin etiquetas corregida'  or  item_dos=='Guardar imagen sin etiquetas corregida':
            height, width, channel = ipaetique.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipaetique, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = QPixmap(qImg.scaled( w,h, Qt.KeepAspectRatio,Qt.SmoothTransformation))
            self.label_3.setPixmap(qImg)
            self.corre.setStyleSheet('QPushButton { background-color: rgb(102, 102, 102);}')
            self.ubicacion.setStyleSheet('QPushButton {background-color: rgb(175, 175, 175);}')

    def closeEvent(self,event):
      reply =QMessageBox.question(self, "Mensaje", "Seguro quiere salir", QMessageBox.Yes, QMessageBox.No)
       
      if reply ==QMessageBox.Yes:
       event.accept()
       import IPython
       app = IPython.Application.instance()
       app.kernel.do_shutdown(True)  
      else: 
       event.ignore()     

    def guardarImagen(self):
        item=self.popup.currentText()
        item_dm=self.popup_DM.currentText()
        item_no=self.popup_NO.currentText()
        item_dos=self.popup_DOS.currentText()
        global name
        
        if item=='Guardar imagen original' or  item_dm=='Guardar imagen original'or  item_no=='Guardar imagen original' or  item_dos=='Guardar imagen original':
            formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
            name = QFileDialog.getSaveFileName(self, "Save as image", "untitled.jpg", formats)
            pixmapImagen_2 = QPixmap(str(filePath))
            pixmapImagen_2.save(str(name[0]))

        if item=='Guardar imagen original corregida'  or  item_dos=='Guardar imagen original corregida':
            formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
            name = QFileDialog.getSaveFileName(self, "Save as image", "untitled.jpg", formats)
            height, width, channel = ipaOri.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipaOri, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmapImagen_2= QPixmap(qImg)
            pixmapImagen_2.save(str(name[0]))
            
        if item=='Guardar imagen sin etiquetas corregida'  or  item_dos=='Guardar imagen sin etiquetas corregida':
            formats = "JPEG (*.jpg;*.jpeg;*jpe;*jfif);;PNG(*.png)"
            name = QFileDialog.getSaveFileName(self, "Save as image", "untitled.jpg", formats)
            height, width, channel = ipaetique.shape
            bytesPerLine = 3 * width
            qImg = QImage(ipaetique, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmapImagen_2= QPixmap(qImg)
            pixmapImagen_2.save(str(name[0]))
        
        if item=='Guardar mascara binaria de RE'or  item_dos=='Guardar mascara binaria de RE':
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
            if should_delete == QMessageBox.Yes:
               from os import remove
               remove(str(filePath))
           
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    