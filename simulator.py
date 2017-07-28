import random

hitBoxLength=10
sceneSize=(400,800)

maxBirdSpeed=10
birdSpeedDescent=0.8
tubeSpeed=3

minTubeInterval=100
minTubeGap=60

class FlySimulator:
    def __init__(self):
        self.birdPos=[sceneSize[1]/2,sceneSize[0]/2]
        self.birdSpeed=0
        self.tubes=[]

        self.frameCount=0
        self.isDead=False

    def reset(self):
        self.__init__()

    def updateBird(self,fly):
        if fly==True:
            self.birdSpeed=maxBirdSpeed
        self.birdPos[1]+=self.birdSpeed
        self.birdSpeed-=birdSpeedDescent

    def tryAddTube(self):
        if len(self.tubes)!=0 and sceneSize[1]-self.tubes[len(self.tubes)-1][0]<minTubeInterval:
            return

        if random.randint(0,100)>5:
            return

        length1=random.randint(0,sceneSize[0]-2*minTubeGap)
        gap=random.randint(minTubeGap,2*minTubeGap)

        if (sceneSize[0]-length1-gap)<0:
            return

        self.tubes.append([sceneSize[1],length1,length1+gap])

    def updateTube(self):
        for tube in self.tubes:
            tube[0]-=tubeSpeed
        if len(self.tubes)!=0 and self.tubes[0][0]<0:
            self.tubes.pop(0)
        self.tryAddTube()

    def isHit(self):
        if self.birdPos[1]-hitBoxLength/2<0 or self.birdPos[1]+hitBoxLength/2>sceneSize[0]:
            self.isDead=True
            return
        for tube in self.tubes:
            if tube[0]>(self.birdPos[0]+hitBoxLength/2) or tube[0]<(self.birdPos[0]-hitBoxLength/2):
                continue
            if tube[1]>(self.birdPos[1]-hitBoxLength/2) or tube[2]<(self.birdPos[1]+hitBoxLength/2):
                self.isDead=True
                return

    def getState(self):
        return [self.isDead,self.birdPos,self.tubes,self.birdSpeed]

    def getLiveTime(self):
        return self.frameCount

    def perform(self,fly):
        if self.isDead!=True:
            self.frameCount+=1
            self.updateBird(fly)
            self.updateTube()
            self.isHit()
        return self.getState()
