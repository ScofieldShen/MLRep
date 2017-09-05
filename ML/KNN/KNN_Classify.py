# __*__ coding: utf-8
from numpy import *
import operator
# 参数：测试集、训练集、标签、k值
def knn_classify(testSet, trainSet, labels, k):
    # 统计训练样本数
    countTrainSet = trainSet.shape[0]

    # 计算测试样本与训练样本之间的距离
    # 先将测试样本复制成与训练样本数相同的数据然后与训练集做差求距离
    minusSample = tile(testSet, (countTrainSet,1))-trainSet
    # 求每个维度距离的平方
    squartminusSample = minusSample**2
    # 按行求和
    plusminusSample = squartminusSample.sum(axis=1)
    # 开方
    extractminusSample = plusminusSample**0.5
    # 对距离按照从小到达排序
    seqasc = extractminusSample.argsort()
    Countlabel={}
    for i in range(k):
        # 选择k个最近距离的样本对距离从小到大排序对应的标签
        ascentLabel = labels[seqasc[i]]
        # 返回样本数量最多的分类作为测试样本分类标签
        Countlabel[ascentLabel] = Countlabel.get(ascentLabel, 0) + 1
    sortedSampleLabel = sorted(Countlabel.items(), key=operator.itemgetter(1), reverse=True)
        # 排序后第一个标签的个数最多，认为是分类向量归属于此标签
    return sortedSampleLabel[0][0]

#   参数:输入文件目录，读取文件内容并返回样本矩阵和标签列表
def file2Matrix(filename):
    # 打开文件
    fr = open(filename)
    lines = fr.readlines()
    # 统计数据行数
    numberlines = len(lines)
    # 创建一个和统计数据维度相同的0矩阵 是一个文件行数为行，3为列的矩阵
    dataMatrix = zeros((numberlines,3))
    # 创建一个存储标签的列表
    classLabel = []
    index = 0
    for line in lines:
        # 将每行数据通过制表符切分,返回一个列表
        datalist = line.strip().split('\t')
        # 将列表中的前三个元素 分别存储到矩阵的第index行每一列
        dataMatrix[index, :] = datalist[0:3]
        # 列表的最后一个字段被取出来，转成int类型，存储到标签列表中
        classLabel.append(int(datalist[-1]))
        index += 1
    # 返回元素矩阵和标签列表
    return dataMatrix,classLabel

# 数据归一化 参数：元素矩阵
def normalData(dataSet):
   #  0是取距震中的列 找出每一列的最小元素
   minSample = dataSet.min(0)
   # 找出每列的最大元素
   maxSample = dataSet.max(0)
   # 算出最大和最小样本之间的取值范围
   ranges = maxSample - minSample
   # 创建一个与dataSet同维的各个元素值为0的矩阵
   normalMAtrix = zeros(shape(dataSet))
   # 计算dataSet的行数
   m = dataSet.shape[0]
   # 计算dataSet矩阵与最小值的元素组成的矩阵的差
   normalMAtrix = dataSet - tile(minSample, (m, 1))
   # 计算dataSet中的元素与每一列最小值元素的差值 与最小与最大值的差组成的矩阵的商求出归一化元素
   normalMAtrix = normalMAtrix/tile(ranges, (m, 1))
   return normalMAtrix,ranges,minSample

def dataClassifyTest(k):
    # 设置测试样本的比重
    hoRatio = 0.10
    # 通过文件转换成矩阵数据 返回矩阵和标签列表
    dataMatrix,classLabel = file2Matrix('/Users/scofield/PycharmProjects/Cross/dataknn.txt')
    # 通过处理返回归一化的数据矩阵、最大最小值间隔、最小值
    normalMatrix,ranges,minSample = normalData(dataMatrix)
    # # 计算所有样本数
    m = normalMatrix.shape[0]
    # 计算测试样本数
    numTestSample = int(m*hoRatio)
    # 错误数
    errorCount=0
    for i in range(numTestSample):
        classifyResult = knn_classify(normalMatrix[i, :], normalMatrix[numTestSample:m,:], classLabel[numTestSample:m], k)
        print(classifyResult)
        print("the result returned back is: %d, the real answer is : %d" %(classifyResult, classLabel[i]))
        if(classifyResult != classLabel[i]):
            errorCount += 1
        errorRate = errorCount/float(numTestSample)
    print("the error rate is : %f" % errorRate)
    print("errorcount is : %d" % errorCount)
    return errorRate

if __name__ == '__main__':
    dataClassifyTest(7)