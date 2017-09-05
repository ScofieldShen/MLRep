import Util.LoadData
import Util.Plot
from numpy import *
import matplotlib as plt

# 梯度上升
# 输入：训练数据，类别标签
def gradAscent(dataSet, classLabels):
    # 将数据转成矩阵
    dataMat = mat(dataSet)
    labelMat = mat(classLabels).T
    # 统计样本数和维度数
    m,n = shape(dataMat)
    # 设置学习率
    alpha = 0.01
    # 设置最大循环次数
    maxCircl = 500
    #
    weights = ones((n,1))
    for i in range(maxCircl):
        h = sigmoid(dataMat*weights)
        # 计算错误率
        error = labelMat - h
        weights = weights + alpha*dataMat.T*error
    return weights

# 随机梯度上升法，顺序遍历每个样本，更新参数。
# 对样本集只遍历一次
# 输入：训练数据，类别标签
def gradAscent01(dataSet, classLabels):
    m,n = shape(dataSet)
    weights = ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(sum(dataSet[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataSet[i]
    return weights

def gradAscent02(dataSet, classLabels, numIter=200):
    m,n = shape(dataSet)
    weights = ones(n)
    for j in range(m):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataSet[randIndex]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataSet[randIndex]
            delete(dataSet, randIndex)
    return weights

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#***分类函数
#输入：待分类的数据inX,更新好的回归系数
def classifyVector(inX, weights):
	#使用sigmoid函数预测
    prob = sigmoid(sum(inX*weights))
	#概率大于0.5判为第1类，概率小于0.5判为第0类
    if prob > 0.5: return 1.0
    else: return 0.0

def colicHorseTest():
    trainData,trainLabels,testData,numTest = Util.LoadData.loadCollicHorData("/Users/scofield/MLRep/Data/LogisticRegressionhorseColicTraining.txt","/Users/scofield/MLRep/Data/LogisticRegressionhorseColicTest.txt")
    trainWeights = gradAscent02(array(trainData),trainLabels, 2)
    errorCount = 0
    # print("testData[0] ： ", testData[0])
    # print("trainWeights ： ", trainWeights)
    # print("testData : ",(testData[0])*trainWeights )
    # print("trainWeights : ", trainWeights)
    num = len(testData)
    for i in range(num):
        pre = testData[i][21]
        arr = delete(testData[i], 21)
        if int(classifyVector(array(arr), trainWeights)) != int(pre):
            errorCount += 1
    errRate = float(errorCount)/numTest
    return errRate



if __name__ == '__main__':
    dataSet,classLabels = Util.LoadData.loadLogisticData("/Users/scofield/MLRep/Data/LogisticRegressiontestSet.txt")
    # print("dataSet : ",dataSet)
    # print("classLabels : ",classLabels)
    # weights = gradAscent(dataSet, classLabels)
    # print(weights)
    # Util.Plot.plotBestFit(dataSet,classLabels,weights.getA())

    # 随机梯度上升法#求参数,把dataArr先由list类型转换成array类型
    # weights = gradAscent01(array(dataSet), classLabels)
    # Util.Plot.plotBestFit(dataSet, classLabels, weights)

    # 改进的随机梯度上升法：随机遍历样本，alpha逐步减小
    # weights = gradAscent02(array(dataSet), classLabels)
    # Util.Plot.plotBestFit(dataSet, classLabels, weights)

    errRate = colicHorseTest()
    print("errRate : ", errRate)