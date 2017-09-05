# __*__ coding: utf-8
from numpy import *
import matplotlib.pyplot as plt
def loadData():
    # train dataset matrix
    dataMatrix = []
    # feature matrix
    classLabels = []
    # read all lines
    lines = open("/Users/scofield/MLRep/Data/LogisticRegressiontestSet.txt").readlines()
    for line in lines:
        # split line into fields
        fields = line.strip().split("\t")
        # set data into datamatrix
        dataMatrix.append([1.0, float(fields[0]), float(fields[1])])
        # set data into classlabels
        classLabels.append(int(fields[2]))
    return dataMatrix,classLabels

def sigmoid(inx):
    print(inx)
    return 1.0/(1+exp(-inx))

def gradientAscent(xArr,yArr):
    # train dataset matrix
    dataMatrix = mat(xArr)
    # labelMatrix transpose
    labelMatrix = mat(yArr).T
    # m samples and n labels
    m,n = shape(dataMatrix)
    # set the step size
    alpha = 0.001
    # set cycle times
    maxCycle = 500
    # regression parameters
    weights = ones((n,1))
    for i in range(maxCycle):
        # predict values
        predict = sigmoid(dataMatrix*weights)
        # predict error
        error = (labelMatrix - predict)
        # update regression coefficient
        weights = weights + alpha * dataMatrix.T * error
    return weights
def plotFit(weights):
    # load data
    dataMatrix,labelMatrix = loadData()
    # put data in array
    dataArray = array(dataMatrix)
    # count samples
    numSamples = shape(dataArray)[0]
    xcoord1 = []
    ycoord1 = []
    xcoord2 = []
    ycoord2 = []
    # cycle datas and draw graph
    for i in range(numSamples):
        if int(labelMatrix[i]) == 1:
            xcoord1.append(dataArray[i,1])
            ycoord1.append(dataArray[i,2])
        else:
            xcoord2.append(dataArray[i, 1])
            ycoord2.append(dataArray[i, 2])
    pig = plt.figure()
    ax = pig.add_subplot(111)
    ax.scatter(xcoord1,ycoord1,s=30,c='red',marker='s')
    ax.scatter(xcoord2,ycoord2,s=30,c='green')
    # create array step size 0.1
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stoGradientAscent01(dataMatrix, classLabels):
    # count samples and dimensions
    m,n = shape(dataMatrix)
    # set step size
    alpha = 0.01
    # create 1xn array
    weights = ones(n)
    # cycle each sample
    for i in range(m):
        # predict the sample's value
        h = sigmoid(sum(dataMatrix[i]*weights))
        # predict error
        error = classLabels[i] - h
        # update the sample's argument
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stoGradientAscent02(dataMatrix, classLabels, numIter):
    # count samples and dimensions
    m, n = shape(dataMatrix)
    # set step size
    alpha = 0.01
    # create 1xn array
    weights = ones(n)
    for i in range(numIter):
        # cycle each sample
        for i in range(m):
            # predict the sample's value
            h = sigmoid(sum(dataMatrix[i] * weights))
            # predict error
            error = classLabels[i] - h
            # update the sample's argument
            weights = weights + alpha * error * dataMatrix[i]
    return weights

def stoGradientAscent03(dataMatrix, classLabels, numIter):
    # count samples and dimensions
    m, n = shape(dataMatrix)
    # create 1xn array
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        # cycle each sample
        for i in range(m):
            # set step size
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            # predict the sample's value
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # predict error
            error = classLabels[randIndex] - h
            # update the sample's argument
            weights = weights + alpha * error * dataMatrix[randIndex]
            delete(dataIndex, randIndex)
    return weights

def classifyVec(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colictest():
    # load train data
    trainData = open("/Users/scofield/MLRep/Data/LogisticRegressionhorseColicTraining.txt").readlines()
    # load test data
    testData = open("/Users/scofield/MLRep/Data/LogisticRegressionhorseColicTest.txt").readlines()
    trainSet = []
    trainLabels = []
    # add data to array
    for line in trainData:
        currentLine = line.strip().split("\t")
        arr = []
        for i in range(21):
            arr.append(float(currentLine[i]))
        trainSet.append(arr)
        trainLabels.append(float(currentLine[21]))
    # calculate weights
    trainWeights = stoGradientAscent03(array(trainSet),trainLabels,2)
    errorCount = 0.0
    numTestSamp = 0.0
    # load test data
    for line in testData:
        currentLine = line.strip().split("\t")
        numTestSamp += 1.0
        arr = []
        for i in range(21):
            arr.append(float(currentLine[i]))
        if int(classifyVec(array(arr), trainWeights)) != int(currentLine[21]):
            errorCount += 1
    # count error rate
    errorRate = (float(errorCount)/numTestSamp)
    return errorRate

def multiTest():
    numtests = 10
    numerrors = 0.0
    for i in range(numtests):
        numerrors += colictest()
    print("after %d iterations the average error rate is: %f" %(numtests, numerrors/float(numtests)))

if __name__ == '__main__':
    # dataMatrix,classLabels = loadData()
    # weights = gradientAscent(dataMatrix,classLabels)
    # plotFit(weights.getA())
    # weights = stoGradientAscent01(array(dataMatrix),classLabels)
    # plotFit(weights)
    # weights = stoGradientAscent03(array(dataMatrix), classLabels, numIter=200)
    # plotFit(weights)
    multiTest()