import Util.LoadData
import Util.Plot
from numpy import *
# 标准回归函数求回归系数：
#求拟合直线的参数w=(X.T*X).I*X.T*y
#样本特征数据xArr
#样本的目标值yArr
def standRegre(xArr, yArr):
    # 将列表数据转成矩阵
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    # 如果矩阵的行列式等于0则不可以求逆
    if linalg.det(xTx) == 0:
        return
    weight = xTx.I * xMat.T*yMat
    return weight

def getGradientDescentWeight(dataSet, classLabels):
    # 将数据转成矩阵
    dataMat = mat(dataSet)
    labelMat = mat(classLabels).T
    # 统计样本数和特征书
    m,n = shape(dataMat)
    # 权值初始化为1，后面根据样本数据调整
	# 训练结束得到最优权值
    # n行一维的列向量
    weights = ones((n,1))
    # 设置循环次数500次，即训练次数，人为给定
    numCyc = 2
    # 设定迭代的步长alpha
    alpha = 0.01
    # 循环maxCycles次，
	# 每次根据模型输出结果与真实值的误差，调整权值。
    for i in range(numCyc):
        # dataMatrix*weights矩阵的乘法。
        # 事实上包含600次的乘积
        # h为模型给出的一个预测值
        h = dataMat*weights
        # 计算误差，每条记录真实值与预测值之差
        error = h - labelMat
        # 权值调整(未知参数调整)，强制转换为matrix类型
        weights = weights - alpha*dataMat.T*error
    return weights


# 随机梯度下降算法
def getStoGradientDescentWeight(dataSet, classLabels, numCir=100):
    # 将数据转成列表
    dataMat = array(dataSet)
    # 统计样本数、特征数
    m,n = shape(dataSet)
    # 创建一个权重的矩阵元素值为1
    weights = ones(n)
    # 设置学习率为0.01
    alpha = 0.01
    for i in range(numCir):
        # 循环每个样本
        for j in range(m):
            # 求参数theta
            theta = sum(dataMat[j]*weights)
            # 计算误差
            error = theta - classLabels[j]
            # 迭代更新weights
            weights = weights - alpha * error * dataMat[j]
    return mat(weights).T



if __name__ == '__main__':
    dataSet, classLabels = Util.LoadData.loadDLFromOneFile("/Users/scofield/MLRep/Data/linearregression.txt")
    # print("dataSet : ", dataSet)
    # print("classLabels : ", classLabels)

    # 最小二乘法
    # weight = standRegre(dataSet, classLabels)
    # Util.Plot.standPlot(dataSet, classLabels, weight)

    # 梯度下降
    weight = getGradientDescentWeight(dataSet, classLabels)
    Util.Plot.standPlot(dataSet, classLabels, weight)

    # 随机梯度下降
    # weights = getStoGradientDescentWeight(dataSet, classLabels)
    # print("weights : ", weights)
    # Util.Plot.standPlot(dataSet, classLabels, weights)