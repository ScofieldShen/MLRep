# __coding__ utf-8
from numpy import *

def loadExData0():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def loadExData1():
    return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
            [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

def cosSim(inA, inB):
    num = float(inA.T * inB)
    inALength = sqrt(float(inA.T * inA))
    inBLength = sqrt(float(inB.T * inB))
    denom = inALength*inBLength
    return 0.5+0.5*(num/denom)

#基于SVD的评分估计
#输入：dataMat 用户数据
#user：用户编号（行）
#simMeas：相似度计算函数
#item:物品编号（列），用户待预测的物品
def svdEst(dataMat, user, simMeas, item):
    #物品数
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    #SVD分解
    U,Sigma,VT = linalg.svd(dataMat)
    #构建对角矩阵，取前3个奇异值
    #3个额外算出来的，确保总能量>90%
    Sig3 = mat(eye(3)*Sigma[:3])
    #SVD降维，低维空间的物品#.I求逆
    xformedItems = dataMat.T * U[:,:3] * Sig3.I
	#遍历所有物品
    for j in range(n):
	    #用户user对j物品评分
        userRating = dataMat[user,j]
		#若未对物品j评分，即userRating=0，不处理
        if userRating == 0 or j==item: continue
		#在低维空间计算物品j与物品item的相似度
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
		#相似度求和
        simTotal += similarity
		#预测用户user对物品item评分总和
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
	#归一化预测评分
    else: return ratSimTotal/simTotal

#基于物品相似度，计算用户对物品的评分估计值
#输入：dataMat 用户数据
#user：用户编号（行）
#simMeas：相似度计算函数
#item:物品编号（列），用户待预测的物品
def standEst(dataMat, user, simMeas, item):
    # 数据矩阵列，即为物品数
    n = shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0

    # 遍历所有物品
    for j in range(n):
        # 用户user对j物品评分
        userRating = dataMat[user, j]
        # （1）若未对物品j评分，即userRating=0，不处理
        if userRating == 0: continue
        # （2）若对物品j评分：
        # 统计对物品item和物品j都评分的用户编号
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                      dataMat[:, j].A > 0))[0]
        # （2.1）若没有用户同时对物品item和j评分，则两物品间相似度为0
        if len(overLap) == 0:
            similarity = 0
        # （2.2）若有用户同时对物品item和j评分，抽取出来，计算相似度
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # 相似度求和
        simTotal += similarity
        # 预测用户user对物品item评分总和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 归一化预测评分
    else:
        return ratSimTotal / simTotal

#基于物品相似度的推荐
#dataMat：  数据
#user：     用户编号
# N：       选择预测评分最高的N个结果
#simMeas：  相似度计算方法
#estMethod：用户对物品的预测估分方法
def recommend(dataMat, user, N=3, simMeans=cosSim, estMethod=standEst):
    # 找没有被用户user评分的物品
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    # 若都评分则退出，不需要再推荐
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    # 遍历未评分的物品
    for item in unratedItems:
        # 预测用户user对为评分物品item的估分
        estimatedScore = estMethod(dataMat, user, simMeans, item)
        # 存（物品编号，对应估分值）
        itemScores.append((item, estimatedScore))
    # 选择最高的估分结果:从高到低排序
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# *****图像压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    # 读.txt文件，转换为矩阵存储到myMat中
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)

    print
    "****original matrix******"
    # 输出阈值处理的图像
    printMat(myMat, thresh)

    # SVD分解
    U, Sigma, VT = linalg.svd(myMat)
    # 初始化numSV*numSV的零矩阵SigRecon
    SigRecon = mat(zeros((numSV, numSV)))
    # Sigma对角线前numSV个值重构对角矩阵SigRecon
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    # 重构后的图像矩阵
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    # 输出阈值处理后的重构图像
    print(printMat(reconMat, thresh))

#*****方便图像显示打印
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            #阈值>0.8，输出1.
            if float(inMat[i,k]) > thresh:
                print (1),
            #阈值<0.8，输出0.
            else: print (0),
        print ('')

#基于皮尔森相关系数的相似度：0.5+0.5*corrcoef()
def pearsSim(inA,inB):
	#并将值范围[-1,1]归一化到[0,1]
    corrcoefMat=corrcoef(inA,inB,rowvar=0) #2*2的对称矩阵
    #索A引[0][1]或者[1][0]都行
    return 0.5+0.5*corrcoefMat[0][1]

if __name__ == '__main__':
    # 1、test the result of strange value
    # 7x5
    # data = [[1, 1, 1, 0, 0],
    #         [2, 2, 2, 0, 0],
    #         [1, 1, 1, 0, 0],
    #         [5, 5, 5, 0, 0],
    #         [1, 1, 0, 2, 2],
    #         [0, 0, 0, 3, 3],
    #         [0, 0, 0, 1, 1]]
    # dataMat = mat(data)
    # U,sigMa,VT = linalg.svd(dataMat)
    # print("U", U)
    # print("eye(3)",eye(3))
    # print("sigMa[:3]",sigMa[:3])
    # print("U[:,:3]",U[:,:3])
    # print("VT[:3,:]",VT[:3,:])
    # sigma1 = mat(eye(3)*sigMa[:3])
    # data1 = U[:,:3]*sigma1*VT[:3,:]
    # print("data1",data1)

    # 2. The similary base on goods
    # data = loadExData()
    # dataMat = mat(data)
    # U,sigMa,VT = linalg.svd(dataMat)
    # itemScores = recommend(dataMat, 1)
    # print("recommend result : " , itemScores)

    # 3. watch the erergy distribution
    # data = loadExData2()
    # dataMat = mat(data)
    # U,sigMa,VT = linalg.svd(dataMat)
    # energy = sigMa**2
    # print("All energy : " , sum(energy))
    # print("90% energy : ", sum(energy)*0.9)
    # print(sum(energy[:3]))

    # 4. recommend based on svd
    data = loadExData2()
    dataMat = mat(data)
    # itemsScores = recommend(data, 1, estMethod=svdEst)
    itemsScores = recommend(data,1,estMethod=svdEst,simMeas=pearsSim)
    print("itemsScores : ", itemsScores)

    #*****图像压缩
    imgCompress(2)


