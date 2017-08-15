import matplotlib.pyplot as plt
from numpy import *
# 图形化显示标准线性回归结果，包括原始数据集及预测结果
# 散点图表示原始的训练数据集
# 蓝色线是通过标准回归函数预测的值
def standPlot(xArr, yArr, w):
    # 将数据转成矩阵 预测数据带入 y = w*x
    xMat = mat(xArr)
    yMat = mat(yArr)
    # 画图
    fig= plt.figure()
    ax = fig.add_subplot(111)
    # 取xMat的第二列数据转成矩阵取第一个列表的数据
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    #产生xmat的副本，对副本操作，就不会影响原始xmat
    xCopy = xMat.copy()
    #按列排序，第0列都是常数1，按列排序其实只是对第1列特征排序，不影响样本分布
    xCopy.sort(0)
    #排序后，做预测并绘图，防止第1列特征大小顺序混乱，导致绘图混乱。
    yHat = xCopy*w
    #只是绘制第1列特征与y的曲线图，需要对第1列特征（x轴变量）排序防止绘图混乱
    ax.plot(xCopy[:,1], yHat)
    plt.show()
