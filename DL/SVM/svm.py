from sklearn import svm
import numpy as np
import pylab as pl
def abs():
    x = [[2,0], [1,1], [2,3]]
    y = [0, 0, 1]
    clf = svm.SVC(kernel="linear")
    # 构建模型 矩阵数据
    clf.fit(x, y)
    print("clf : ", clf)
    # 支持向量(在同超平面平行的平面上)的点
    print("vectors : ", clf.support_vectors_)
    # 支持向量的角标
    print("support : ", clf.support_)
    # 统计每个域中的支持向量
    # print("n_support : ", clf.n_support_)

def acd():
    # 随机函数随机抓取数据
    np.random.seed(0)
    # 产生20行，2列的数据。
    x = np.r_[np.random.rand(20,2) - [2,2], np.random.rand(20,2) + [2,2]]
    # 产生标签数据
    y = [0]*20 + [1]*20
    # print(x)
    # print(y)
    # 构建模型
    clf = svm.SVC(kernel="linear")
    clf.fit(x, y)
    print(clf)
    # switching to the generic n-dimensional parameterization of the hyperplan to the 2D-specific equation
    # of a line y=a.x +b: the generic w_0x + w_1y +w_3=0 can be rewritten y = -(w_0/w_1) x + (w_3/w_1)
    w = clf.coef_[0]
    # 转成点斜式求斜率
    a = -w[0]/w[1]
    # 在15 到5之间产生数据
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0])/w[1]
    # 找出边际线 斜率相同截距不同
    b = clf.support_vectors_[0]
    # print("b : ", b)
    yy_down = a * xx + (b[1] - a*b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a*b[0])
    # print("yy : ", yy)
    # print("yy_down : ", yy_down)
    # print("yy_up : ", yy_up)
    pl.plot(xx, yy)
    pl.plot(xx, yy_down)
    pl.plot(xx, yy_up)

    pl.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
               s= 80, facecolors='none')
    pl.scatter(x[:,0],x[:,1], c=y, cmap=pl.cm.Paired)

    pl.axis('tight')
    pl.show()

if __name__ == '__main__':
    acd()