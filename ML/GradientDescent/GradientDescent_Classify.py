# __*__ coding:utf-8
from numpy import *
import matplotlib.pyplot as plt
def loadData(filename):
    # count the length of feature dimensions
    numDF = len(open(filename).readline().split("\t"))-1
    # read all lines
    lines = open(filename).readlines()
    dataList = []
    targetList = []
    for line in lines:
        lineList = []
        # split a line into to array
        fields = line.strip().split("\t")
        for i in range(numDF):
            # put feature's dimensation in an array
            lineList.append(float(fields[i]))
        # store all lines in an array
        dataList.append(lineList)
        # store all target values
        targetList.append(float(fields[-1]))
    return dataList,targetList

def standRegre(xArr,yArr):
    xMatrix = mat(xArr)
    yMatrix = mat(yArr).T
    # the transpose of x matrix multiply x matrix it can't be 0
    xTx = xMatrix.T*xMatrix
    if linalg.det(xTx) == 0.0:
        print("this is a singular matrix can not be inverse")
        return
    # return the argument
    ws = xTx.I * xMatrix.T * yMatrix
    return ws

def graidentDescent(dataInArr,classLabelArr):
    # create dimensation matrix
    dataMatrix = mat(dataInArr)
    # create target value matrix
    labelMatrix = mat(classLabelArr).T
    # get samples and domensations
    m,n = shape(dataMatrix)
    # create weight matrix
    weights = ones((n,1))
    # set step size
    α=0.01
    # set loop times
    maxCycle = 500
    for i in range(maxCycle):
        # got feature dimensation matrix
        a = dataMatrix * weights
        # error value
        error = a - labelMatrix
        # get weight
        weights = weights - α * dataMatrix.T * error
    return weights

def stoGradientDescent(dataList, classLabels, numCir=100):
    # create data array
    dataMatrix = array(dataList)
    # count samples and dimensations
    m,n = shape(dataList)
    # set the step value
    α = 0.01
    # set weight
    weights = ones(n)
    # set the circulate times
    for i in range(numCir):
        # circulate each sample
        for j in range(m):
            #
            𝛉 = sum(dataMatrix[j] * weights)
            error = 𝛉 - classLabels[j]
            weights = weights - α * error * dataMatrix[j]
    return  mat(weights).transpose()

def standPlot(xArr, yArrr, weight):
    xMatrix = mat(xArr)
    yMatrix = mat(yArrr)
    # mark pot
    pig = plt.figure()
    ax = pig.add_subplot(111)
    ax.scatter(xMatrix[:,1].flatten().A[0], yMatrix.T[:,0].flatten().A[0])
    xCopy = xMatrix.copy()
    # sort
    xCopy.sort(0)
    # predicted values
    yhat = xCopy*weights
    ax.plot(xCopy[:,1],yhat)
    plt.show()

if __name__ == '__main__':
    dataList,targetList = loadData("/Users/scofield/MLRep/Data/gradientdescent.txt")
    # print(dataList,targetList)
    # 最小二乘法
    # 梯度下降
    # 随机梯度下降
    weights = stoGradientDescent(dataList,targetList)
    # print(weights)
    standPlot(dataList, targetList, weights)