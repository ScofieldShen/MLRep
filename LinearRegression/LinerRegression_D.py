#
from numpy import *
import matplotlib.pyplot as plt
# dataSet 训练数据特征值列表
# classLabels 训练数据标签列表
def loadData(fileName):
    # 用于存储特征值
    dataSet = []
    # 用于存储标签值
    classLabels = []
    # 维度数
    numDimensations = len(open(fileName).readline().split("\t")) - 1
    # 读取所有数据
    lines = open(fileName).readlines()
    # 将每一行数据放入到列表
    for line in lines:
        # 存储每行特征数据
        strList = []
        # 切分每行数据、去除空格
        lineStr = line.strip().split("\t")
        for i in range(numDimensations):
            # 将每个字段转成float类型
            strList.append(float(lineStr[i]))
        dataSet.append(strList)
        # 将所有的标签值转成float存储
        classLabels.append(float(lineStr[-1]))
    return dataSet,classLabels

# 线性回归参数公式w = (X.T*X).I * X.T*Y
# dataArr 特征值
# labelArr 标签值
def standRegression(dataArr, labelArr):
    # 将数据转换成矩阵
    XMat= mat(dataArr)
    # 将数据转成矩阵并 将行转列
    YMat= mat(labelArr).T
    #
    xTx = XMat.T * XMat
    # 要对xTx求逆 行列式不可以为0
    if linalg.det(xTx) == 0:
        return
    # 求回归参数
    ws = xTx.I * XMat.T*YMat
    return ws

# 图形化显示标准线性回归结果，包括原始数据集及预测结果
# 散点图表示原始的训练数据集
# 蓝色线是通过标准回归函数预测的值
def standPlot(dataArr, labelArr, w):
    # 将数据转成matrix 预测数据带入公式 y = x*w
    xMat = mat(dataArr)
    yMat = mat(labelArr)
    # 画点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #绘散点图，目标值y与第1维度的特征的坐标关系
    #flatten将array或matrix中的数据展开得到1xn的矩阵形式。
    #.A表示将1xn的matrix转换成1xn的array，.A[0]取这一行数据（array类型）
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    #产生xmat的副本，对副本操作，就不会影响原始xmat
    xCopy = xMat.copy()
    #按列排序，第0列都是常数1，按列排序其实只是对第1列特征排序，不影响样本分布
    xCopy.sort(0)
    #排序后，做预测并绘图，防止第1列特征大小顺序混乱，导致绘图混乱。
    yHat = xCopy*w #预测值
    #只是绘制第1列特征与y的曲线图，需要对第1列特征（x轴变量）排序防止绘图混乱
    ax.plot(xCopy[:,1],yHat)
    plt.show()

# 局部加权线性回归
# testPoint 测试点数据
# XArr 训练集特征数据
# yArr 训练集标签数据
def lwlr(testPoint, xArr, yArr, k=1.0):
    # 将特征数据转换成矩阵
    xMat = mat(xArr)
    # 将标签数据转成矩阵 将行转为列
    yMat = mat(yArr).T
    # 统计样本数
    m = shape(xMat)[0]
    # 创建一个mxm的方阵
    weights = mat(eye(m))
    # 循环每个样本，计算每个样本与测试样本点之间的权值
    for i in range(m):
        # 测试点与训练样本之间的向量差
        diffMat = testPoint - xMat[i,:]
        # 对角矩阵weights对角线上存权值alpha
        # print("diffMat",diffMat)
        # print("diffMat.T", diffMat.T)
        weights[i,i] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * weights * xMat
    # 如果行列式为0 xTx补课取逆
    if linalg.det(xTx) == 0:
        return
    # 求回归系数
    ws = xTx.I * xMat.T*weights*yMat
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def lwlrPlot(testPredict, xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    ax.plot(xSort[:, 1], testPredict[srtInd])
    plt.show()


if __name__ == '__main__':
    dataSet,classLabels = loadData("/Users/scofield/MLRep/Data/linearregression.txt")
    ws = standRegression(dataSet, classLabels)
    # print("Ws : ", ws)
    # standPlot(dataSet, classLabels, ws)
    # testpre = lwlr(dataSet[0], dataSet, classLabels, 1)
    # print("testpre : ", testpre)
    predict = lwlrTest(dataSet, dataSet, classLabels, k=0.01)
    lwlrPlot(predict, dataSet, classLabels)