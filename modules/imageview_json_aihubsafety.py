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
import json
import random
from glob import glob


class ImageViewer(QMainWindow):
    def __init__(self,save_dir):
        super(ImageViewer, self).__init__()

        # jsons6= glob("../../datasets/aihub_safety/val/**/2.중장비만/*.json") 
        # jsons5= glob("../../datasets/aihub_safety/val/*/2.중장비만/*.json") 
        jsons4= glob("../../datasets/aihub_safety/val/*/1.안전보호구만/*.json") 
        jsons3= glob("../../datasets/aihub_safety/val/*/5.전체/*.json") 
        jsons2= glob("../../datasets/aihub_safety/val/*/*/1.안전보호구만/*.json") 
        jsons= glob("../../datasets/aihub_safety/val/*/*/5.전체/*.json") 
        # jsons = jsons + jsons2 + jsons3 + jsons4 + jsons5 + jsons6
        jsons = jsons + jsons2 + jsons3 + jsons4 

        random.seed(42)
        random.shuffle(jsons)

        self.gray_color_table = [qRgb(i, i, i) for i in range(256)]

        self.img_list = [js_f.replace('라벨','원천').replace('/5.전체','').replace('/2.중장비만','').replace('/1.안전보호구만','').replace('.json','.jpg') for js_f in jsons]
        self.base_path = self.img_list
        self.label_list = jsons

        self.saved_label_file = ''

        self.pos = 0
        self.total = len(self.img_list)
        self.save_dir = save_dir
        
        self.printer = QPrinter()
        self.width = 1920
        self.height = 1080
        
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
        if not os.path.exists(os.path.join(self.save_dir , 'images')):
            os.mkdir(os.path.join(self.save_dir , 'images'))
        
        if not os.path.exists(os.path.join(self.save_dir , 'labels')):
            os.mkdir(os.path.join(self.save_dir , 'labels'))
        
        image = cv2.imread(self.img_list[self.pos])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(jsons[self.pos],"r") as f:
            data = json.load(f)
        self.annos = data['annotations']
        img_size = data['image']['resolution']
        clses = []
        colors = {}


            
        self.new_data = []
        self.image_size = [image.shape[1],image.shape[0]]
        
        for anno in self.annos:
            
            line = ''
            # if anno['class'] not in ['01','02','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']:
            #     continue
            if int(anno['class']) > 8:
                continue
            
            cls = anno['class']


            if cls not in clses:
                clses.append(cls)
                colors[cls] = tuple([random.randint(0,255) for _ in range(3)])
            try:
                xyxy = anno['box'] 
            except:
                continue

            line += str(int(cls) -1) + ' ' + ' '.join(str(i) for i in self.cvt_yoloformat(xyxy,self.image_size[1],self.image_size[0])) + '\n'
            self.new_data.append(line)

            image = cv2.rectangle(image , pt1=(xyxy[0],xyxy[1]),pt2=(xyxy[2],xyxy[3]),color= colors[cls], thickness=3)
            image = cv2.putText(img=image, text=cls, org= (xyxy[0],xyxy[1]),fontFace=cv2.FONT_HERSHEY_PLAIN ,fontScale=4,color=(0,255,0),thickness=5)
        

        
        self.width = image.shape[1]
        self.height = image.shape[0]
        
        self.filename = os.path.basename(self.img_list[self.pos])
        self.setWindowTitle("Image Viewer"+ self.filename + '[%d / %d]'%(self.pos, len(self.img_list)))
        
        # print(self.img_list[self.pos])
        # labels = self.read_label(self.img_list[self.pos])
        # for label in labels:
        #     label = label.strip().split(' ')
        #     # if int(label[0]) == 0:
        #     pt1 = (int((float(label[1]) - float(label[3])/2) * self.width) , int((float(label[2])- float(label[4])/2) * self.height) ) 
        #     pt2 = (int((float(label[1]) + float(label[3])/2) * self.width) , int((float(label[2]) + float(label[4])/2) * self.height) ) 
        #     image = cv2.rectangle(image , pt1=pt1 , pt2=pt2 , color=(255,0,0), thickness=2)
        #     image = cv2.putText(img=image, text=label[0], org= pt1,fontFace=cv2.FONT_HERSHEY_PLAIN ,fontScale=4,color=(0,255,0),thickness=5)
        
        self.openImage(image=self.toQImage(image))
        

    def cvt_yoloformat(self,xyxy,w,h):
        box_w = xyxy[2] - xyxy[0]
        box_h = xyxy[3] - xyxy[1]
        return [(xyxy[0] + box_w/2)/w ,(xyxy[1] + box_h/2)/h, box_w/w, box_h/h ]

    def read_label(self,image_path):
        label_path = os.path.dirname(image_path).split('/images')[0] +'/labels/'+ os.path.basename(image_path)[:-4] + '.txt'
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
        if e.key() == 65:
            if not self.pos == 0:
                self.pos -= 1
                image = cv2.imread(self.img_list[self.pos])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.width = image.shape[1]
                self.height = image.shape[0]
                self.filename = os.path.basename(self.img_list[self.pos])
                self.setWindowTitle("Image Viewer"+ self.filename + '[%d / %d]'%(self.pos, len(self.img_list)))
                with open(self.label_list[self.pos],"r") as f:
                    data = json.load(f)
                self.annos = data['annotations']
                clses = []
                colors = {}

                self.new_data = []
                self.image_size = [image.shape[1],image.shape[0]]
                
                for anno in self.annos:
                    
                    line = ''
                    # if anno['class'] not in ['01','02','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']:
                    #     continue
                    if int(anno['class']) > 8:
                        continue
                    
                    cls = anno['class']


                    if cls not in clses:
                        clses.append(cls)
                        colors[cls] = tuple([random.randint(0,255) for _ in range(3)])
                    try:
                        xyxy = anno['box'] 
                    except:
                        continue

                    line += str(int(cls) -1) + ' ' + ' '.join(str(i) for i in self.cvt_yoloformat(xyxy,self.image_size[1],self.image_size[0])) + '\n'
                    self.new_data.append(line)

                    image = cv2.rectangle(image , pt1=(xyxy[0],xyxy[1]),pt2=(xyxy[2],xyxy[3]),color= colors[cls], thickness=3)
                    image = cv2.putText(img=image, text=cls, org= (xyxy[0],xyxy[1]),fontFace=cv2.FONT_HERSHEY_PLAIN ,fontScale=4,color=(0,255,0),thickness=5)
                
                
                """
                이미지 처리
                """
                self.openImage(image=self.toQImage(image))
                print('\r' + self.img_list[self.pos], end="")
                                                
        elif e.key() == 68:
            self.pos += 1
            if self.total == self.pos:
                self.pos -= 1
            image = cv2.imread(self.img_list[self.pos])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.width = image.shape[1]
            self.height = image.shape[0]
            self.filename = os.path.basename(self.img_list[self.pos])
            self.setWindowTitle("Image Viewer"+ self.filename + '[%d / %d]'%(self.pos, len(self.img_list)))

            with open(self.label_list[self.pos],"r") as f:
                data = json.load(f)
            self.annos = data['annotations']
            clses = []
            colors = {}        
            self.new_data = []
            self.image_size = [image.shape[1],image.shape[0]]
            
            for anno in self.annos:
                
                line = ''
                # if anno['class'] not in ['01','02','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']:
                #     continue
                if int(anno['class']) > 8:
                    continue
                
                cls = anno['class']


                if cls not in clses:
                    clses.append(cls)
                    colors[cls] = tuple([random.randint(0,255) for _ in range(3)])
                try:
                    xyxy = anno['box'] 
                except:
                    continue

                line += str(int(cls) -1) + ' ' + ' '.join(str(i) for i in self.cvt_yoloformat(xyxy,self.image_size[1],self.image_size[0])) + '\n'
                self.new_data.append(line)

                image = cv2.rectangle(image , pt1=(xyxy[0],xyxy[1]),pt2=(xyxy[2],xyxy[3]),color= colors[cls], thickness=3)
                image = cv2.putText(img=image, text=cls, org= (xyxy[0],xyxy[1]),fontFace=cv2.FONT_HERSHEY_PLAIN ,fontScale=4,color=(0,255,0),thickness=5)
            
            """
            이미지 처리
            """
            
            self.openImage(image=self.toQImage(image))            
            print('\r' + self.img_list[self.pos], end="")
        
        elif e.key() == 80:
            print('copy')
            image = cv2.imread(self.img_list[self.pos])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(os.path.join(self.save_dir + '/images'))
            shutil.copy(self.img_list[self.pos] , os.path.join(self.save_dir + '/images'))

            self.saved_label_file = os.path.join(self.save_dir + '/labels') + '/'+ os.path.basename(self.label_list[self.pos])[:-4] + 'txt'
            
            with open(self.saved_label_file, 'w' ) as f:
                for asd in self.new_data:
                    f.write(asd)


            # shutil.copy(self.label_list[self.pos] ,os.path.join(self.save_dir + '/labels'))
            
            self.openImage(image=self.toQImage((cv2.putText(img=image , text='copied',color=(255,0,0),org=(int(self.width/2.3),self.height//4), fontScale=1,thickness=4,fontFace=cv2.FONT_HERSHEY_COMPLEX))))
        
        elif e.key() == 73:
            print('delete')
            image = cv2.imread(self.img_list[self.pos])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            saved_label_file = os.path.join(self.save_dir + '/labels') + '/'+ os.path.basename(self.label_list[self.pos])[:-4] + 'txt'
            try:
                os.remove(os.path.join(self.save_dir + '/images') +'/' + os.path.basename(self.img_list[self.pos] ) )
                os.remove(saved_label_file)
                self.openImage(image=self.toQImage((cv2.putText(img=image , text='delete',color=(255,0,0),org=(int(self.width/2.3),self.height//4), fontScale=1,thickness=4,fontFace=cv2.FONT_HERSHEY_COMPLEX))))
            except:
                self.openImage(image=self.toQImage((cv2.putText(img=image , text='no_file',color=(0,255,0),org=(int(self.width/2.3),self.height//4), fontScale=1,thickness=4,fontFace=cv2.FONT_HERSHEY_COMPLEX))))
                print('no deleting file')

            
            # self.openImage(image=self.toQImage(image))            p
        
if __name__ == '__main__':

    import sys
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--img-path', type=str , default= '.')
    # # # parser.add_argument('--label-path', type=str , default= '.')
    # # parser.add_argument('--save-path', type=str , default= '.')
    # # opt = parser.parse_args()

    app = QApplication(sys.argv)
    imageViewer = ImageViewer('../../datasets/aihub_safety/temp')
    imageViewer.show()
    sys.exit(app.exec_())