from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, qRgb
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QInputDialog)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import glob
import os
import cv2
import numpy as np
import shutil
import time
import argparse
import pickle


class ImageViewer(QMainWindow):
    def __init__(self,imgs,save_dir,scale):
        super(ImageViewer, self).__init__()

        if os.path.exists('./log.txt'):
            with open('./log.txt', 'r') as f:
                self.lognum = [int(i) for i in f.readline().split(' ')]
        else:
            with open('./log.txt', 'w') as f:
                f.write('0 0')
                self.lognum = [0,0]

        self.prekey = ''

        self.gray_color_table = [qRgb(i, i, i) for i in range(256)]
        self.base_path = imgs

        self.img_list = glob.glob(os.path.join(self.base_path, 'images/*.jpg'))

        self.label_list = [os.path.dirname(image_path).split('/images')[0] +'/labels/'+ os.path.basename(image_path)[:-4] + '.txt' for image_path in self.img_list]
        
        self.img_list.sort()
        self.label_list.sort()
        self.pos = self.lognum[0] - self.lognum[1] 
        if self.pos >= len(self.img_list):
            self.pos =len(self.img_list) -1


        self.total = len(self.img_list)
        self.save_dir = save_dir
        
        self.printer = QPrinter()
        self.width = int(960 * scale)
        self.height = int(540 * scale)

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
    
        self.setWindowTitle("Image Viewer")
        self.resize(self.width, self.height)

        # print('dir ---------- ' + os.path.join(self.save_dir , 'images'))
        if not os.path.exists(os.path.join(self.save_dir , 'filtered_images')):
            print(self.save_dir)
            print(os.path.join(self.save_dir , 'filtered_images'))
            os.mkdir(os.path.join(self.save_dir , 'filtered_images'))
        
        if not os.path.exists(os.path.join(self.save_dir , 'filtered_labels')):
            os.mkdir(os.path.join(self.save_dir , 'filtered_labels'))
        
        image = cv2.imread(self.img_list[self.pos])
        none_count = 0
        while True:
            if none_count > len(self.img_list):
                return 0
            if image is None:
                self.pos  += 1
                none_count += 1
                image = cv2.imread(self.img_list[self.pos])
            else:
                break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        self.filename = os.path.basename(self.img_list[self.pos])
        self.setWindowTitle("Image Viewer"+ self.filename + '[%d / %d]'%(self.pos, len(self.img_list)))
        
        print(self.img_list[self.pos])
        labels = self.read_label(self.img_list[self.pos])
        for label in labels:
            label = label.strip().split(' ')
            # if int(label[0]) == 0:
            pt1 = (int((float(label[1]) - float(label[3])/2) * image.shape[1]) , int((float(label[2])- float(label[4])/2) * image.shape[0]) ) 
            pt2 = (int((float(label[1]) + float(label[3])/2) * image.shape[1]) , int((float(label[2]) + float(label[4])/2) * image.shape[0]) ) 
            image = cv2.rectangle(image , pt1=pt1 , pt2=pt2 , color=(255,0,0), thickness=2)
            image = cv2.putText(img=image, text=label[0], org= pt1,fontFace=cv2.FONT_HERSHEY_PLAIN ,fontScale=4,color=(0,255,0),thickness=5)
        


        self.openImage(image=self.toQImage(cv2.resize(image,(self.width,self.height))))
        # self.openImage(image=self.toQImage(image))
        
    def read_label(self,image_path):
        label_path = os.path.dirname(image_path).split('images')[0] +'labels/'+ os.path.basename(image_path)[:-4] + '.txt'
        with open(label_path, 'r') as f:
            labels = f.readlines()
        return labels
        
    
    def normalSize(self):
        self.imageLabel.adjustSize()

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()
        self.updateActions()
        
    def createActions(self):
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)
        
    def updateActions(self):
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())
       
    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))
        
    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(self.gray_color_table)
                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim
                
    def openImage(self, image=None, fileName=None):
            if image == None:
                image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))

            self.fitToWindowAct.setEnabled(True)
            self.updateActions()
            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()
                
    def keyPressEvent(self, e):

        ## a
        if e.key() == 65:
            if not self.pos == 0:
                self.lognum[0] -= 1
                if self.prekey == 'i':
                    self.lognum[0] += 1

                with open('./log.txt', 'w') as f:
                    f.write(str(self.lognum[0]) + ' ' + str(self.lognum[1]))
                
                self.pos -= 1
                image = cv2.imread(self.img_list[self.pos])
                none_count = 0
                while True:
                    if self.pos < 0:
                        break
                    if image is None:
                        self.pos  -= 1
                        none_count += 1
                        image = cv2.imread(self.img_list[self.pos])
                    else:
                        break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.filename = os.path.basename(self.img_list[self.pos])
                self.setWindowTitle("Image Viewer"+ self.filename + '[%d / %d]'%(self.pos, len(self.img_list)))
                labels = self.read_label(self.img_list[self.pos])
                for label in labels:
                    label = label.strip().split(' ')
                    # if int(label[0]) == 0:
                    pt1 = (int((float(label[1]) - float(label[3])/2) * image.shape[1]) , int((float(label[2])- float(label[4])/2) * image.shape[0]) ) 
                    pt2 = (int((float(label[1]) + float(label[3])/2) * image.shape[1]) , int((float(label[2]) + float(label[4])/2) * image.shape[0]) ) 
                    image = cv2.rectangle(image , pt1=pt1 , pt2=pt2 , color=(255,0,0), thickness=2)
                    image = cv2.putText(img=image, text=label[0], org= pt1,fontFace=cv2.FONT_HERSHEY_PLAIN ,fontScale=4,color=(0,255,0),thickness=5)
                """
                이미지 처리
                """
                print('...')
                self.openImage(image=self.toQImage(cv2.resize(image,(self.width,self.height))))
                # self.openImage(image=self.toQImage(image))
                print('\r' + self.img_list[self.pos], end="")
                
                self.prekey = 'a'

        ## d                                                
        elif e.key() == 68:
            if self.total != self.pos:
                self.pos += 1
                self.lognum[0] += 1
                with open('./log.txt', 'w') as f:
                    f.write(str(self.lognum[0]) + ' ' + str(self.lognum[1]))
                image = cv2.imread(self.img_list[self.pos])
                none_count = 0
                while True:
                    if none_count > len(self.img_list):
                        return 0
                    if image is None:
                        self.pos  += 1
                        none_count += 1
                        image = cv2.imread(self.img_list[self.pos])
                    else:
                        break            
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.filename = os.path.basename(self.img_list[self.pos])
                self.setWindowTitle("Image Viewer"+ self.filename + '[%d / %d]'%(self.pos, len(self.img_list)))
                labels = self.read_label(self.img_list[self.pos])
                for label in labels:
                    label = label.strip().split(' ')
                    # if int(label[0]) == 0:
                    pt1 = (int((float(label[1]) - float(label[3])/2) * image.shape[1]) , int((float(label[2])- float(label[4])/2) *image.shape[0]) ) 
                    pt2 = (int((float(label[1]) + float(label[3])/2) * image.shape[1]) , int((float(label[2]) + float(label[4])/2) * image.shape[0]) ) 
                    image = cv2.rectangle(image , pt1=pt1 , pt2=pt2 , color=(255,0,0), thickness=2)
                    image = cv2.putText(img=image, text=label[0], org= pt1,fontFace=cv2.FONT_HERSHEY_PLAIN ,fontScale=4,color=(0,255,0),thickness=5)
                """
                이미지 처리
                """
                
                self.openImage(image=self.toQImage(cv2.resize(image,(self.width,self.height))))
                # self.openImage(image=self.toQImage(image))            
                print('\r' + self.img_list[self.pos], end="")
                self.prekey = 'd'
        # elif e.key() == 80:
        #     print('copy')
        #     image = cv2.imread(self.img_list[self.pos])

        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     print(os.path.join(self.save_dir + '/images'))
        #     shutil.copy(self.img_list[self.pos] , os.path.join(self.save_dir + '/images'))
        #     shutil.copy(self.label_list[self.pos] ,os.path.join(self.save_dir + '/labels'))
            
        #     self.openImage(image=self.toQImage((cv2.putText(img=image , text='copied',color=(255,0,0),org=(int(self.width/2.3),self.height//4), fontScale=1,thickness=4,fontFace=cv2.FONT_HERSHEY_COMPLEX))))
        elif e.key() == 73: ## i
            print('try moving')

            self.lognum[1] += 1
            with open('./log.txt', 'w') as f:
                f.write(str(self.lognum[0]) + ' ' + str(self.lognum[1]))
            image = cv2.imread(self.img_list[self.pos])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(os.path.join(self.save_dir + '/images') +'/' + os.path.basename(self.img_list[self.pos] ) )
            try:
                print(os.path.join(self.save_dir + 'filtered_images'))
                print(os.path.join(self.save_dir + 'filtered_images'))
                shutil.move(self.img_list[self.pos] ,os.path.join(self.save_dir + '/filtered_images'))
                shutil.move( self.img_list[self.pos].replace('images','labels')[:-3] + 'txt' ,os.path.join(self.save_dir + '/filtered_labels')  )

                self.openImage(image=self.toQImage(cv2.resize((cv2.putText(img=image , text='moved',color=(255,0,0),org=(int(image.shape[1]/2.3),image.shape[0]//4), fontScale=1,thickness=4,fontFace=cv2.FONT_HERSHEY_COMPLEX)),(self.width,self.height))))
            except:
                self.openImage(image=self.toQImage(cv2.resize((cv2.putText(img=image , text='no_file',color=(0,255,0),org=(int(image.shape[1]/2.3),image.shape[0]//4), fontScale=1,thickness=4,fontFace=cv2.FONT_HERSHEY_COMPLEX)),(self.width,self.height))))
                print('failed. not exist moving file')

            self.prekey = 'i'
            
            # self.openImage(image=self.toQImage(image))            p
        
if __name__ == '__main__':

    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str , default= '.')
    parser.add_argument('--save-path', type=str , default= '.')
    parser.add_argument('--scale', type=float , default= 1)
    opt = parser.parse_args()

    app = QApplication(sys.argv)
    # print(opt.img_path)
    imageViewer = ImageViewer(opt.img_path, opt.save_path,opt.scale)
    imageViewer.show()
    sys.exit(app.exec_())
