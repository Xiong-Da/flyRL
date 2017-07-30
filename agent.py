import threading
import random
import copy

import tensorflow as tf
import numpy as np

import simulator

#[speed,birdPosY,tube-1X,tube-1height,tube-1gap,tube0...,tube1...]
input_state=tf.placeholder(tf.float32,[None,1+1+3*3])
#[flyActionValue,notFlyActionValue]
input_value=tf.placeholder(tf.float32,[None])

weight1 = tf.Variable(tf.truncated_normal([1+1+3*3,128], stddev=0.1))
bias1 = tf.Variable(tf.zeros([128]))
output1=tf.nn.relu(tf.matmul(input_state,weight1)+bias1)

weight2 = tf.Variable(tf.truncated_normal([128,128], stddev=0.1))
bias2 = tf.Variable(tf.zeros([128]))
output2=tf.nn.relu(tf.matmul(output1,weight2)+bias2)

weight3 = tf.Variable(tf.truncated_normal([128,2], stddev=0.1))
bias3 = tf.Variable(tf.zeros([2]))
output3=tf.matmul(output2,weight3)+bias3

value_array=output3

input_action=tf.placeholder(tf.float32,[None,2])
output_value=tf.reduce_sum(tf.multiply(value_array,input_action),axis=1)

diff_vector=input_value-output_value

loss=tf.reduce_mean(tf.square(diff_vector))
train_op=tf.train.AdamOptimizer(0.0005).minimize(loss)

initer=tf.global_variables_initializer()
saver=tf.train.Saver()

##############################################################

def addTubeData(data,tube):
    data.append(tube[0])
    data.append(tube[1])
    data.append(tube[2]-tube[1])

def translateState(rawSate):
    retList=[rawSate[3],rawSate[1][1]]

    birdPos=rawSate[1]
    tubes=rawSate[2]

    closestTube=None
    for tube in tubes:
        if tube[0]<birdPos[0]-simulator.hitBoxLength/2:
            continue
        closestTube=tube
        break

    maxTubeGap=simulator.sceneSize[0]
    if closestTube==None:
        return [rawSate[3],rawSate[1][1],
                birdPos[0]-100,0,maxTubeGap,birdPos[0]+100,0,maxTubeGap,birdPos[0]+200,0,maxTubeGap]

    index=tubes.index(closestTube)
    if index==0:
        retList.append(birdPos[0]-100)
        retList.append(0)
        retList.append(maxTubeGap)
    else:
        addTubeData(retList,tubes[index-1])
    addTubeData(retList,tubes[index])
    if index==(len(tubes)-1):
        addTubeData(retList,tubes[index])
        retList[len(tubes)-3]+=100
    else:
        addTubeData(retList,tubes[index+1])

    return retList

def getValues(sess,state):
    values=sess.run([value_array],feed_dict={input_state:state})[0]
    return values

def makeDecision(value):
    if value[0][0]>=value[0][1]:
        return True
    return False

def makeGreedyDecision(value,rate):
    if random.uniform(0,1)>rate:
        return makeDecision(value)
    if random.uniform(0,1)>0.9:
        return True
    else:
        return False

playDataValue=[]
playDataState=[]
playDataAction=[]
maxPlayListLen=1024*32
def updatePlayDataList(sess,actions,rawOldStates,rawNewStates):
    global playDataValue,playDataState,playDataAction
    rewards=[]
    newStates=[]
    oldStates=[]

    for state in rawNewStates:
        newStates.append(translateState(state))
        if state[0]==True:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    for state in rawOldStates:
        oldStates.append(translateState(state))

    values = np.max(getValues(sess, newStates), axis=1) * 0.99 + rewards

    for i in range(len(values)):
        if rewards[i] <= 0.1:
            values[i] = 0

    if len(playDataValue)<maxPlayListLen:
        for i in range(len(actions)):
            playDataValue.append(values[i])
            playDataAction.append(actions[i])
            playDataState.append(oldStates[i])
    else:
        for i in range(len(actions)):
            index=random.randint(0,len(playDataValue)-1)
            playDataValue[index]=values[i]
            playDataAction[index]=actions[i]
            playDataState[index]=oldStates[i]


def getBatch(num):
    states=[]
    actions=[]
    values=[]

    for i in range(num):
        index=random.randint(0,len(playDataValue)-1)
        states.append(playDataState[index])
        actions.append(playDataAction[index])
        values.append(playDataValue[index])

    return states,actions,np.array(values)


###############################################################

isStarted=False
savePath=None
message=""

def getMessage():
    return message

def startTrainExistMoudle(path):
    global isStarted
    if isStarted == True:
        return
    isStarted = True
    thread = threading.Thread(target=trainThreadFun, args=[tf.get_default_graph(),path])
    thread.setDaemon(True)
    thread.start()

def startTrain():
    startTrainExistMoudle(None)

def setPath(path):
    global savePath
    savePath=path

lastPlayCount=50
greedyFactor=0.1
def playForTrainData(sess,flySimu):
    global lastPlayCount,greedyFactor
    actions=[]
    rawOldStates=[]
    rawNewStates=[]

    playCount=0
    liveCount=0
    clickCount=0
    while True:
        if len(actions)>=1024:
            break

        playCount+=1

        _actions=[]
        _rawNewStates=[]
        _rawOldStates=[]
        flySimu.reset()
        while True:
            oldState=flySimu.getState()
            dec=makeGreedyDecision(
                getValues(sess,[translateState(oldState)]),greedyFactor)
            if dec==True:
                action=[1,0]
                clickCount+=1
            else:
                action=[0,1]
            newState=flySimu.perform(dec)

            _actions.append(action)
            _rawNewStates.append(newState)
            _rawOldStates.append(oldState)

            if newState[0]==True:
                liveCount+=flySimu.getLiveTime()
                recordLength=40
                if len(_actions)>=recordLength and random.uniform(0,1)<=0.5:
                    actions=actions+_actions[-recordLength:]
                    rawNewStates=rawNewStates+_rawNewStates[-recordLength:]
                    rawOldStates=rawOldStates+_rawOldStates[-recordLength:]
                else:
                    actions = actions + _actions
                    rawNewStates = rawNewStates + _rawNewStates
                    rawOldStates = rawOldStates + _rawOldStates
                break

    greedyFactor=(200-liveCount/playCount)*0.001
    if greedyFactor<0.01:
        greedyFactor=0.01

    lastPlayCount=playCount
    print("playCount:"+str(playCount)+" perLiveTime:"+str(int(liveCount/playCount))+
          " perClick:"+str(int(clickCount/playCount))+" greedyFactor:"+str(greedyFactor))

    updatePlayDataList(sess,actions,rawOldStates,rawNewStates)

def trainThreadFun(graph,path):
    global message,savePath,isStarted

    sess = tf.Session(graph=graph)
    sess.run([initer])

    if path!=None:
        saver.restore(sess=sess, save_path=path)

    flySimu=simulator.FlySimulator()

    iterCount=0
    while savePath==None:
        playForTrainData(sess,flySimu)

        for i in range(4):
            iterCount += 1
            oldStates,actions,values=getBatch(64)
            _,curLoss,curDiff=sess.run([train_op,loss,diff_vector],
                                    feed_dict={input_state:oldStates,input_action:actions,input_value:values})
            message="iter count:"+str(iterCount)
            if iterCount%100==0:
                print("\ncurDiff:")
                print(curDiff)
                print("\ntrainValue:")
                print(values)
                print("\ncurrent loss:"+str(curLoss)+"\n")

    isStarted=False
    saver.save(sess, savePath)
    sess.close()
    savePath=None
    message=""

###############################################################

playSess=None

def selectModule(path):
    global playSess
    if playSess!=None:
        playSess.close()
    playSess=tf.Session()
    saver.restore(sess=playSess, save_path=path)

def flyOrNot(state):
    if playSess==None:
        raise Exception("play session is not inited,please select a moudle")

    return makeDecision(getValues(playSess,[translateState(state)]))