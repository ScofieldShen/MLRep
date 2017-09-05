from numpy import *
import operator
def loadData(fileName):
    # 读取数据
    lines = open(fileName).readlines()
    # 获取样本数
    m = len(lines)
    # 创建一个m行3列的矩阵
    outData = zeros((m,3))
    # 创建列表存储标签数据
    classLabels = []
    index = 0
    for line in lines:
        # 将每一行去掉空格并切分
        strLine = line.strip().split("\t")
        # 将切分的数据存储到矩阵中
        outData[index,:] = strLine[0:3]
        # 取最后一列 转成int型将标签数据存储到列表
        classLabels.append(int(strLine[-1]))
        index += 1
    return outData,classLabels

def normalizeData(dataSet):
    # 求每一列的最小值
    minVals = dataSet.min(0)
    # 求每一列的最大值
    maxVals = dataSet.max(0)
    # 获取样本数
    m = shape(dataSet)[0]
    # 计算最大最小值间隔
    rangeVals = maxVals - minVals
    # 数据集与最小数据集差值
    outData = zeros(shape(dataSet))
    outData = dataSet - tile(minVals, (m, 1))
    # 求归一化数据
    outData = outData/tile(rangeVals, (m, 1))
    return outData

def knnClassify(testData, trainData, labels, k):
    # 统计样本数
    m = shape(trainData)[0]
    # 创建测试数据的矩阵
    testMat = tile(testData, (m, 1))
    # 求差
    minusData = trainData - testMat
    # 求平方
    squareData = minusData**2
    # 求和
    sumData = squareData.sum(axis=1)
    # 求开根号
    sqrtData = sumData**0.5
    # 排序 从小到大返回角标
    sortedid = sqrtData.argsort()
    countLabel = {}
    for i in range(k):
        ascentL = labels[sortedid[i]]
        countLabel[ascentL] = countLabel.get(ascentL, 0) + 1
    endLabel = sorted(countLabel.items(), key=operator.itemgetter(1), reverse=True)
    return endLabel[0][0]

def testKnnClassify(k):
    # 测试数据百分比
    hRatio = 0.1
    # 加载数据
    dataMat, labels = loadData("/Users/scofield/MLRep/Data/dataknn.txt")
    # 统计样本数
    m = shape(dataMat)[0]
    # 统计测试数据数目
    numTest = int(m*hRatio)
    # 对数据进行归一化操作
    dataSet = normalizeData(dataMat)
    # 统计错误数
    errcount = 0
    for i in range(numTest):
        # 测试数据就是从0到 numTest 训练呢集是从numTest到最后 训练标签窃取numTest到m 得到预测结果
        resultLabel = knnClassify(dataSet[i], dataSet[numTest:m,:], labels[numTest:m], k)
        # 判断预测结果是否准确 如果错误则在统计的错误数中+1
        if resultLabel != labels[i]:
            errcount += 1
    # 计算错误率
    errrorRate = errcount/float(numTest)
    return errrorRate

if __name__ == '__main__':
    # dataMat,labels = loadData("/Users/scofield/MLRep/Data/dataknn.txt")
    # print("dataMat",dataMat)
    # print("labels",labels)
    # outData= normalizeData(dataMat)
    # print("outData",outData)
    k = 4
    error = testKnnClassify(k)
    print("error : ", error)