import sys
import os
from PyQt5.QtWidgets import qApp, QAction, QApplication, QMainWindow,QWidget,QHBoxLayout
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon,QPainter,QColor,QBrush
from PyQt5.QtCore import Qt,pyqtSignal,QTimer

import simulator
import agent

class FlyWidget(QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self.data=None

        self.shrinkFactor=1.0
        self.moveFactor=0.0

    def setData(self,data):
        self.data=data
        self.update()

    def updateTransformFactor(self):
        geometry = self.geometry()
        self.shrinkFactor=geometry.height()/simulator.sceneSize[0]
        self.moveFactor=(geometry.width()-simulator.sceneSize[1]*self.shrinkFactor)/2

    def transfromPoints(self,points):
        retPoints=[]
        for point in points:
            retPoints.append([point[0]*self.shrinkFactor+self.moveFactor,
                              self.geometry().height()-point[1]*self.shrinkFactor])
        return retPoints

    def paintEvent(self, e):
        super().paintEvent(e)

        data=self.data
        if data==None:
            return

        painter = QPainter(self)
        painter.setPen(Qt.blue)
        painter.setBrush(Qt.blue)

        self.updateTransformFactor()
        birdPos=data[0]
        tubes=data[1]

        halfHitBox = simulator.hitBoxLength / 2
        birdPos = [[birdPos[0] - halfHitBox, birdPos[1] + halfHitBox],
                   [birdPos[0] + halfHitBox, birdPos[1] - halfHitBox]]
        birdPos = self.transfromPoints(birdPos)
        painter.drawRect(birdPos[0][0], birdPos[0][1],
                                  birdPos[1][0] - birdPos[0][0], birdPos[1][1] - birdPos[0][1])

        painter.setPen(Qt.red)

        for tube in tubes:
            points=[]
            points.append([tube[0],0])
            points.append([tube[0],tube[1]])
            points.append([tube[0],tube[2]])
            points.append([tube[0],simulator.sceneSize[0]])

            points=self.transfromPoints(points)

            painter.drawLine(points[0][0],points[0][1],points[1][0],points[1][1])
            painter.drawLine(points[2][0],points[2][1],points[3][0],points[3][1])

class FlyWindow(QMainWindow):
    def __init__(self,parent):
        super().__init__(parent)

        self.title="flyRL"
        self.isClicked=False
        self.isPlay=False
        self.isShow=False
        self.flySimulator=simulator.FlySimulator()

        self.flyWidget = FlyWidget(None)

        self.playAction=QAction("play",self)
        self.trainAction=QAction("train",self)
        self.saveAction=QAction("save",self)
        self.selectAction=QAction("select",self)
        self.showAction=QAction("show",self)

        self.timer = QTimer(self)

        self.initUI()
        self.initAction()

    def initUI(self):
        layout = QHBoxLayout(None)
        layout.addWidget(self.flyWidget)
        widget=QWidget(None)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        menubar = self.menuBar()

        optionMenu=menubar.addMenu("options")
        optionMenu.addAction(self.playAction)
        optionMenu.addAction(self.trainAction)
        optionMenu.addAction(self.saveAction)
        optionMenu.addAction(self.selectAction)
        optionMenu.addAction(self.showAction)

        self.resize(400,400)

    def initAction(self):
        self.playAction.triggered.connect(self.onPlay)
        self.trainAction.triggered.connect(self.onTrain)
        self.saveAction.triggered.connect(self.onSave)
        self.selectAction.triggered.connect(self.onSelect)
        self.showAction.triggered.connect(self.onShow)

        self.timer.timeout.connect(self.onTimer)
        self.timer.start(50)

    def onPlay(self):
        self.isClicked=False
        self.isShow=False
        self.isPlay=True
        self.flySimulator.reset()

    def onTrain(self):
        agent.startTrain()

    def onSave(self):
        path = QFileDialog.getSaveFileName(self, "save path")[0]
        if len(path) == 0: return
        print(path)
        agent.setPath(path)

    def onSelect(self):
        path = QFileDialog.getOpenFileName(self, "select")[0]
        if len(path) == 0: return
        path = os.path.splitext(path)[0]
        print(path)
        agent.selectModule(path)

    def onShow(self):
        self.isPlay=False
        self.isShow=True
        self.flySimulator.reset()

    def onTimer(self):
        message=agent.getMessage()
        if len(message)!=0:
            self.setWindowTitle(message)
        else:
            self.setWindowTitle(self.title)

        if self.isPlay==True:
            ret=self.flySimulator.perform(self.isClicked)
            self.isClicked=False
            self.flyWidget.setData([ret[1],ret[2]])
        elif self.isShow==True:
            ret=self.flySimulator.perform(agent.flyOrNot(self.flySimulator.getState()))
            self.flyWidget.setData([ret[1],ret[2]])

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.isClicked=True

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F5:
            self.flySimulator.reset()

if __name__ =="__main__":
    app = QApplication(sys.argv)
    window = FlyWindow(None)
    window.show()
    sys.exit(app.exec_())