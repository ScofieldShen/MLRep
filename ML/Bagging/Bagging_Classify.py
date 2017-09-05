#基本分类器为逻辑回归,数据集：马疝病
#ratio=0.8, bagnum=21 ,错误率0.31,0.32,0.26,...
#ratio=1.0, bagnum=21 ,错误率0.26,0.23,0.28,...
#原始的逻辑回归分类，错误率：0.35左右
from numpy import *

def loadData(fileName):
    fr = open(fileName)
    numDimen = len(fr.readline().split("\t"))
    dataMatrix = []
    labelMatrix = []
    for line in fr.readlines():
        arrFea = []
        arrLine = line.strip().split("\t")
        for i in range(numDimen-1):
            arrFea.append(float(arrLine[i]))
        dataMatrix.append(arrFea)
        labelMatrix.append(float(arrLine[-1]))
    return dataMatrix,labelMatrix

def sigmoid(inx):
    return 1.0/(1+exp(-inx))

#***改进的随机梯度上升法
#随机遍历每个样本，更新参数；
#循环多次遍历样本集
#输入：训练数据dataMatrix，类别标签classLabels，默认迭代次数numIter
def sortGradientAscent(dataMatrix, classLabels, numIter):
    # print("dataMatrix",dataMatrix)
    # print("classLabels",classLabels)
    # m 训练样本个数 n 特征维数
    m,n = shape(dataMatrix)
    # 回归系数，1 x n行向量，n个特征对应n个回归系数 init 1
    weights= ones(n)
    # loop numIter times
    for j in range(numIter):
        # create index 0 ~ m-1
        dataIndex = range(m)
        # loop each sample
        for i in range(m):
            # update steps
            alpha = 4.0/(i+j+1.0)+0.01
            # choose an index
            randIndex = int(random.uniform(0,len(dataIndex)))
            # model predict value
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            # calculate predict error
            error = classLabels[randIndex] - h
            # update Regression confiction
            weights = weights + alpha * error * dataMatrix[randIndex]
            # delete sample which was used to update confiction
            delete(dataMatrix, randIndex)
    return weights

#***分类函数
#输入：待分类的数据inX,更新好的回归系数
def classifyVector(inx, weights):
    # use sigmoid function predict
    prob = sigmoid(sum(inx*weights))
    # if prob bigger than 0.5 classify to 1
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# LR errorrate PRedict
def LRPredict(dataSet, classLabels, testData, testLabel):
    # calculate weights
    weights = sortGradientAscent(dataSet, classLabels, numIter=200)
    # count error number
    errorCount = 0
    for i in range(len(testData)):
        if int(classifyVector(testData[i,:], weights)) != testLabel[i]:
            errorCount += 1
    # calculate error rate
    errorRateLR = (float(errorCount)/len(testLabel))
    return errorRateLR

#有放回的采样
#dataSet：数据集（不含类别标签）
#labels：类别标签
#bagCapacity：每次有放回的抽取样本数
def baggingSample(dataSet, labels, bagCapacity):
    randIndex = []
    for i in range(bagCapacity):
        index = int(random.uniform(0, bagCapacity))
        randIndex.append(index)
    sampleData = dataSet[randIndex,:]
    sampleLabel = labels[randIndex]
    return sampleData,sampleLabel

#bagging方法得到多个分类器，预测多个结果进行投票表决
def majorityCnt(labelList):
    # create dictionary the number of samples : classify
    items = dict([((labelList.count(i),i))for i in labelList])
    # return the samples of the most classify
    return items[max(items.keys())]

def baggingLRPredict(dataSet, labelList, testData, testLabel, sampleRatio, bagNum):
    bagCapacity = int(len(dataSet)*sampleRatio)
    numTest = len(testLabel)
    predictTestArr  = zeros((numTest, bagNum))
    for i in range(bagNum):
        bagData,bagLabel = baggingSample(dataSet, labelList, bagCapacity)
        weights = sortGradientAscent(bagData, bagLabel, numIter=100)
        for j in range(numTest):
            predictTestArr[j,i] = int(classifyVector(testData[j,:], weights))
    errCount = 0
    for j in range(numTest):
        if majorityCnt(list(predictTestArr[j,:])) != testLabel[j]:
            errCount +=1
    errRate = float(errCount)/numTest
    return errRate,predictTestArr


if __name__ == '__main__':
    dataSet,labelMatrix = loadData("/Users/scofield/MLRep/Data/horseColicTraining.txt")
    testData,testLabel = loadData("/Users/scofield/MLRep/Data/horseColicTest.txt")
    dataSet = array(dataSet)
    labelMatrix = array(labelMatrix)
    testData = array(testData)
    testLabel = array(testLabel)
    errorRateLR= []
    # for i in range(5):
    #     errorRate = LRPredict(dataSet, labelMatrix, testData, testLabel)
    #     errorRateLR.append(errorRate)
    # print("All error rate of LR classification is : " , errorRate)
    # print("Average of LR classification : " , sum(errorRate)/5)

    errorRateAll=[]
    for i in range(5):
        sampleRatio = 1.0
        bagNum = 21
        errorRate,pretestArr = baggingLRPredict(dataSet, labelMatrix, testData, testLabel, sampleRatio, bagNum)
        errorRateAll.append(errorRate)
    print("All error rate of LR classification is : " , errorRate)
    print("Average of LR classification : " , sum(errorRateAll)/5)