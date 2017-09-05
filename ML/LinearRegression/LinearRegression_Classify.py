#__*__ coding:utf-8
from numpy import *
import matplotlib.pyplot as plt
def loadData(filename):
    # count the dimension of the features
    numDF = len(open(filename).readline().split("\t"))-1
    lines = open(filename).readlines()
    # store the dimension of features
    dataList = []
    # store the target value
    targetList = []
    # loop each line and store the value
    for line in lines:
        # split each line and the store a line in a list
        fields = line.strip().split("\t")
        lineList = []
        for i in range(numDF):
            lineList.append(float(fields[i]))
        dataList.append(lineList)
        # choose the last field of a line then store it in a line
        targetList.append(float(fields[-1]))
    return dataList,targetList

def standRegre(xArr, yArr):
    # create x matrix
    xMatrix = mat(xArr)
    # create y matrix translate row to rank
    yMatrix = mat(yArr).T
    xtx = xMatrix.T*xMatrix
    # determinate can not be 0 so it can be inverse
    if linalg.det(xtx)==0.0:
        print("this is singular matrix can not be matrix ")
        return
    # the linear regression function
    ws = xtx.I * xMatrix.T * yMatrix
    return ws

def standplot(xArr, yArr, w):
    # create matrix store features
    xmatrix = mat(xArr)
    # create matrix store target value
    ymatrix = mat(yArr).T
    # create a page
    fg = plt.figure()
    # create x,y dimension
    ax = fg.add_subplot(111)
    # flatten translate an array or matrix to an matrix（1*n）
    # .A translate matrix to array
    ax.scatter(xmatrix[:,1].flatten().A[0],ymatrix[:,0].flatten().A[0])
    xcp = xmatrix.copy()
    # sort by the first line
    xcp.sort(0)
    # predict value
    yhat = xcp*w
    ax.plot(xcp[:,1],yhat)
    plt.show()

def lwlr(testPoint,xArr,yArr,k=1.0):
    xmatrix = mat(xArr)
    ymatrix = mat(yArr).T
    numsample = shape(xmatrix)[0]
    # create a m*m matrix
    weights = mat(eye(numsample))
    for i in range(numsample):
        diffmatrix = testPoint - xmatrix[i,:]
        weights[i,i] = exp(diffmatrix*diffmatrix.T/(-2.0*k**2))
    xTx = xmatrix.T*weights*xmatrix
    if linalg.det(xTx)==0.0:
        print("this is a singular matrix can't be inverse")
        return
    # return the regression argument
    ws = xTx.I*(xmatrix.T*weights*ymatrix)
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    numSample = shape(testArr)[0]
    yhat = zeros(numSample)
    for i in range(numSample):
        yhat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yhat

def lwlrplot(testPredict, xArr, yArr):
    xmatrix = mat(xArr)
    ymatrix = mat(yArr).T
    pit = plt.figure()
    ax = pit.add_subplot(111)
    ax.scatter(xmatrix[:,1].flatten().A[0],ymatrix[:,0].flatten().A[0],s=2)
    sortid= xmatrix[:,1].argsort(0)
    xsort = xmatrix[sortid][:,0,:]
    ax.plot(xsort[:,1],testPredict[sortid])
    plt.show()


if __name__ == '__main__':
    data,real = loadData("/Users/scofield/MLRep/Data/linearregression.txt")
    # print(data,real)
    es = standRegre(data,real)
    # print("es=", es)
    # standplot(data,real,es)
    # targetpredict = lwlr(data[0],data,real,1)
    # targetpredict = lwlrTest(data,data,real,k=0.01)
    # lwlrplot(targetpredict,data,real)