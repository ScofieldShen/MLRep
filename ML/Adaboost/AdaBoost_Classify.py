# __*__ coding: utf-8
from numpy import *
def loadSampleData():
    datMatrix = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMatrix,classLabels

#dataMatrix：样本数据
#dimen：特征编号，将该列特征进行阈值比较对数据分类
#threshVal：阈值
#threshIneq：两种阈值选择：‘lt’：小于阈值的预测类别值为-1，‘gt’：大于阈值的预测类别取-1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    #单层决策树，根据与阈值比较，返回类别
    retArray = ones((shape(dataMatrix)[0],1))
    #小于阈值，取类别-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    #大于阈值，取类别-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

# *****构建单层决策分类树，adaboost每次迭代过程中的基分类器
#找最佳的切分点（第几列特征，具体取值，如何划分）
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
	#训练数据个数m,特征维数n
    m,n = shape(dataMatrix)
	#某个特征值范围内，递增选阈值，numSteps为所选阈值总数
    numSteps = 10.0
	#字典，存储在给定样本权值D下的决策树
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))#最佳预测类别
    minError = inf #初始化误差率为无穷
    #遍历所有特征维度
    for i in range(n):
        #求当前特征的范围值，从小到大按一定步长递增选择阈值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps #步长
        #遍历整个特征值范围，递增取步长（多取了两个值j=-1,j=int(numSteps)+1）
        for j in range(-1,int(numSteps)+1):
            #两种阈值划分：'lt'表示小于阈值取-1，大于阈值取+1；‘gt’与‘lt’相反
            for inequal in ['lt', 'gt']:
                #按步长单调递增取阈值
                threshVal = (rangeMin + float(j) * stepSize)
                #利用单层决策树预测类别
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                #统计预测情况：正确为1，错误为0
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
				#分类误差率：错分样本的权值之和
                weightedError = D.T*errArr
                # print 'split: dim %d, thresh %.2f,inequal %s, weights error %.4f'\
                # %(i,threshVal,inequal,weightedError)

                #求最小的分类误差率下的决策树分类器
                if weightedError < minError:
                    minError = weightedError
					#基分类器的预测结果
                    bestClasEst = predictedVals.copy() #备份预测值
                    #最佳单层决策树的信息
                    bestStump['dim'] = i #划分维度
                    bestStump['thresh'] = threshVal#划分阈值
                    bestStump['ineq'] = inequal #划分方式,取'lt'or'gt'
    return bestStump,minError,bestClasEst

def simpDataShow(dataMat,classLabels):
    import matplotlib.pyplot as plt
    xcord1=[]
    ycord1=[]
    xcord0=[]
    ycord0=[]

    for i in range(len(classLabels)):
        if classLabels[i]==1.0:
            #类别为1的样本的第0维特征，作为x轴
            xcord1.append(dataMat[i,0])
            #类别为1的样本的第1维特征，作为y轴
            ycord1.append(dataMat[i,1])
        else:#类别为-1的样本的第0，1列特征
            xcord0.append(dataMat[i,0])
            ycord0.append(dataMat[i,1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,marker='s',s=60,c='green')
    ax.scatter(xcord1,ycord1,marker='o',s=50,c='red')
    plt.show()

#*****加载数据
def loadDataSet(fileName):
    #特征数（包含最后一列的类别标签）
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
	#按行读取
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        #把每个样本特征存入 dataMat中
        dataMat.append(lineArr)
        #存每个样本的标签
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#*****测试算法性能
#datToClass：待分类的测试数据
#classifierArr：训练出的集成分类器
def adaClassify(datToClass,classifierArr):
    #测试样本的特征属性（不含标签）
    dataMatrix = mat(datToClass)
	#测试样本个数
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    #多个分类器预测值，进行加权求和
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
	#最终预测的类别
    return sign(aggClassEst)


# *****adaboost算法
# dataArr：训练样本数据
# classLabels：训练样本的类别
# numIt：迭代次数
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    # 存每次迭代中生成的弱分类器
    weakClassArr = []
    # 训练样本数
    m = shape(dataArr)[0]

    # step1:初始化权值分布
    D = mat(ones((m, 1)) / m)
    # 存每次迭代后的累计估计值
    aggClassEst = mat(zeros((m, 1)))

    # step2：迭代
    for i in range(numIt):
        # step2.1-2.2:基本弱分类器及分类误差率
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # step2.3:计算alpha
        # max(error,1e-16)为了防止error=0时除以0溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 存最佳分类器的权重alpha
        bestStump['alpha'] = alpha
        # 将各个基分类器的相关信息存在数组中
        weakClassArr.append(bestStump)
        # 打印预测的类别
        # step2.4:更新样本权值D
        # 计算更新权值的指数部分 #multiply是矩阵相应元素相乘
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        # 更新样本权值D
        D = multiply(D, exp(expon))
        D = D / D.sum()  # 归一化
        # step3：构建本次迭代后的各个基分类器的线性组合f(x)
        # 本次迭代的累计估计值之和
        aggClassEst += alpha * classEst
        # step4：本次迭代后的分类器G(x)=sign(f(x))
        # 本次迭代后训练样本的分类情况：错误标1，正确标0,#计算分类错误率：错误数/总数
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        # 本轮迭代后集成的分类错误率
        errorRate = aggErrors.sum() / m
        # 当分类错误率为0，停止迭代
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst, errorRate

if __name__ == '__main__':
    # dataMatrix,classLabels = loadSampleData()
    # print("dataMatrix",dataMatrix)
    # print("classLabels",classLabels)
    # #初始化样本权重
    # D=mat(ones((len(classLabels),1))/len(classLabels))
    # bestStump,minError,bestClasEst=buildStump(dataMatrix,classLabels,D)
    # print (bestStump)
    # print (minError)
    # print (bestClasEst)
    # #显示数据分布
    # simpDataShow(dataMatrix,classLabels)
    # classifierArray,aggClassEst,classLabels=adaBoostTrainDS(datamat,classlabels,9)
    # print ('classifierArray:',classifierArray)
    # # # #对测试样本进行预测类别标签
    # result=adaClassify([[0,0],[3,4],[5,5]],classifierArray)
    # print ('result=',result)

    #-------------------马疝病数据集horseColicTraining2.txt修改了horseColicTraining.txt的类别标签
    dataArr, labelArr = loadDataSet('/Users/scofield/MLRep/Data/horseColicTraining.txt')
    classifierArray, aggClassEst, traing_err_rate = adaBoostTrainDS(dataArr, labelArr, 10)
    #马疝病测试分类器性能
    testArr, testlabelArr = loadDataSet('/Users/scofield/MLRep/Data/horseColicTest.txt')
    predictlabel = adaClassify(testArr, classifierArray)  # 预测标签
    # 计算测试数据的分类错误率
    test_num = len(testlabelArr)
    errArr = mat(ones((test_num, 1)))
    err_num = errArr[predictlabel != mat(testlabelArr).T].sum()
    err_rate = err_num / test_num
    print('test_err_rate=', err_rate)
    print('traing_err_rate=', traing_err_rate)