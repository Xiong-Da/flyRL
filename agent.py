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

weight1 = tf.Variable(tf.truncated_normal([1+1+3*3,32], stddev=0.1))
bias1 = tf.Variable(tf.zeros([32]))
output1=tf.nn.relu(tf.matmul(input_state,weight1)+bias1)

weight2 = tf.Variable(tf.truncated_normal([32,64], stddev=0.1))
bias2 = tf.Variable(tf.zeros([64]))
output2=tf.nn.relu(tf.matmul(output1,weight2)+bias2)

weight3 = tf.Variable(tf.truncated_normal([64,64], stddev=0.1))
bias3 = tf.Variable(tf.zeros([64]))
output3=tf.nn.relu(tf.matmul(output2,weight3)+bias3)

weight4 = tf.Variable(tf.truncated_normal([64,32], stddev=0.1))
bias4 = tf.Variable(tf.zeros([32]))
output4=tf.nn.relu(tf.matmul(output3,weight4)+bias4)

weight5 = tf.Variable(tf.truncated_normal([32,2], stddev=0.1))
bias5 = tf.Variable(tf.zeros([2]))
output5=tf.nn.relu(tf.matmul(output4,weight5)+bias5)

value_array=output5

input_action=tf.placeholder(tf.float32,[None,2])
output_value=tf.reduce_sum(tf.multiply(value_array,input_action),axis=1)

diff_vector=input_value-output_value

loss=tf.reduce_mean(tf.square(diff_vector))
train_op=tf.train.AdamOptimizer(0.0001).minimize(loss)

initer=tf.global_variables_initializer()
saver=tf.train.Saver()

##############################################################

def addTubeData(data,birdPos,tube):
    data.append(tube[0]-birdPos[0])
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
    if closestTube==None:
        return [rawSate[3],rawSate[1][1],
                -100,0,simulator.sceneSize[1],100,0,simulator.sceneSize[1],200,0,simulator.sceneSize[1]]

    index=tubes.index(closestTube)
    if index==0:
        retList.append(-100)
        retList.append(0)
        retList.append(simulator.sceneSize[1])
    else:
        addTubeData(retList,birdPos,tubes[index-1])
    addTubeData(retList,birdPos,tubes[index])
    if index==(len(tubes)-1):
        addTubeData(retList,birdPos,tubes[index])
        retList[len(tubes)-2]+=100
    else:
        addTubeData(retList,birdPos,tubes[index+1])

    return retList

def getValues(sess,state):
    values=sess.run([value_array],feed_dict={input_state:state})[0]
    return values

def makeDecision(value):
    if value[0][0]>=value[0][1]:
        return True
    return False

def makeGreedyDecision(value,rate):
    if random.randint(0,1000)>1000*rate:
        return makeDecision(value)
    if random.randint(0,100)>50:
        return True
    else:
        return False

deadDataList=[]
maxDeadListLen=256
numOfDeadDataAddEachIter=5
def updateDeadList(data):
    if len(deadDataList)<maxDeadListLen:
        deadDataList.append(data)
    else:
        deadDataList[random.randint(0,maxDeadListLen-1)]=data

def addDeadData(actions,newStates,oldStates):
    if len(deadDataList)<=numOfDeadDataAddEachIter:
        return
    for i in range(numOfDeadDataAddEachIter):
        index=random.randint(0,len(deadDataList)-1)
        actions.append(deadDataList[index][0])
        newStates.append(deadDataList[index][1])
        oldStates.append(deadDataList[index][2])

def getTrainData(sess,actions,rawOldStates,rawNewStates):
    rewards=[]
    newStates=[]
    oldStates=[]

    _actions=[]
    _rawOldStates=[]
    _rawNewStates=[]

    for index in range(len(rawOldStates)):
        if rawNewStates[index][0]==True:
            updateDeadList([actions[index],rawNewStates[index],rawOldStates[index]])

    for i in range(50):
        index=random.randint(0,len(actions)-1)
        _actions.append(actions[index])
        _rawNewStates.append(rawNewStates[index])
        _rawOldStates.append(rawOldStates[index])

    addDeadData(_actions,_rawNewStates,_rawOldStates)

    for state in _rawNewStates:
        newStates.append(translateState(state))
        if state[0]==True:
            rewards.append(0)
        else:
            rewards.append(1)
    for state in _rawOldStates:
        oldStates.append(translateState(state))

    values=np.max(getValues(sess,newStates),axis=1)*0.9+rewards

    for i in range(len(values)):
        if rewards[i]==0:
            values[i]=0

    return oldStates,_actions,values

###############################################################

isStarted=False
savePath=None
message=""

def getMessage():
    return message

def startTrain():
    global isStarted
    if isStarted==True:
        return
    isStarted=True
    thread = threading.Thread(target=trainThreadFun, args=[tf.get_default_graph()])
    thread.setDaemon(True)
    thread.start()

def setPath(path):
    global savePath
    savePath=path

def playForTrainData(sess,flySimu):
    actions=[]
    rawOldStates=[]
    rawNewStates=[]

    playCount=0
    clickCount=1
    while True:
        if len(actions)>=1000:
            break

        tempActions=[]
        tempOldStates=[]
        tempNewStates=[]
        playCount+=1
        flySimu.reset()
        while True:
            oldState=flySimu.getState()
            dec=makeGreedyDecision(getValues(sess,[translateState(oldState)]),0.05)
            if dec==True:
                action=[1,0]
                clickCount+=1
            else:
                action=[0,1]
            newState=flySimu.perform(dec)

            tempActions.append(copy.deepcopy(action))
            tempNewStates.append(copy.deepcopy(newState))
            tempOldStates.append(copy.deepcopy(oldState))

            if newState[0]==True or len(tempActions)>=5000:
                if len(tempActions)>50:
                    actions+=tempActions[-50:]
                    rawOldStates+=tempOldStates[-50:]
                    rawNewStates+=tempNewStates[-50:]
                else:
                    actions+=tempActions
                    rawOldStates+=tempOldStates
                    rawNewStates+=tempNewStates
                break
    print("playCount:"+str(playCount)+" clickCount:"+str(clickCount)+" per:"+str(clickCount/playCount))
    return getTrainData(sess,actions,rawOldStates,rawNewStates)

def trainThreadFun(graph):
    global message,savePath,isStarted

    sess = tf.Session(graph=graph)
    sess.run([initer])

    flySimu=simulator.FlySimulator()

    iterCount=0
    while savePath==None:
        iterCount+=1
        oldStates,actions,values=playForTrainData(sess,flySimu)
        _,curLoss,curDiff=sess.run([train_op,loss,diff_vector],feed_dict={input_state:oldStates,input_action:actions,
                                       input_value:values})
        message="iter count:"+str(iterCount)
        if iterCount%10==0:
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