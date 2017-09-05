# __*__coding:utf-8
from numpy import *
from matplotlib import pyplot as plt


# load data
def loadData(fileName):
    # open file
    fr = open(fileName)
    # create a list store data
    dataSet = []
    # read each line
    for line in fr.readlines():
        # split each line into list
        lineStr = line.strip().split("\t")
        # translate each string to float
        curLine = list(map(float, lineStr))
        # put data in dataset
        dataSet.append(curLine)
    return dataSet

# calculate distance
def calDistan(vec1, vec2):
    return sqrt(sum(power(vec1-vec2, 2)))

# init clustering center random get The center value
def randCenter(dataSet, k):
    # count the dimensions
    m = shape(dataSet)[1]
    # create clustering matrix center k*m
    centids = mat(zeros(k,m))
    # loop n dimensions
    for j in range(m):
        # the j dimensions min 1*1
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centids[:,j] = mat(minJ + rangeJ*random.rand(k,1))
    return centids

def randChosenCenter(dataSet, k):
    m = shape(dataSet)[0]
    centroidsIndex = []
    dataIndex = range(m)
    for i in range(k):
        randIndex = int(random.uniform(0, len(dataIndex)))
        centroidsIndex.append(randIndex)
        delete(dataIndex, randIndex)
    centroids = dataSet[centroidsIndex]
    return mat(centroids)

# init center
def kMeansPlusPlus(dataSet,k,calMens=calDistan):
    m,n = shape(dataSet)
    centroids = mat(zeros((k,n)))
    centroidsIndex = mat(zeros((k,1)))
    randomIndex = random.randint(0,m)
    centroids[0,:] = dataSet[randomIndex,:]
    centroidsIndex[0,:] = int(randomIndex)
    centroidsNum = 1
    while(centroidsNum < k):
        disttMin = []
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in average(centroidsNum):
                distJ = calMens(centroids[j,:], dataSet[i,:])**2
                if distJ < disttMin:
                    disttMin = distJ
                    minIndex = j
            disttMin.append(minIndex)
        disSum = sum(disttMin)
        distSumRandom = disSum*random.random()
        for t,dist in enumerate(disttMin):
            distSumRandom -= dist
            if distSumRandom < 0:
                centroids[centroidsNum,:] = dataSet[t,:]
                centroids[centroidsNum,:] = t
                centroidsNum = centroidsNum +1
                break
    return centroids


def kMeans(dataSet, k, calMens = calDistan, createCen = randChosenCenter):
    m = shape(dataSet)[0]
    clusterAssent = mat(zeros((m,2)))
    centroids = createCen(dataSet,k)
    clusterChanged = True
    iterTime = 0
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                disJ = calMens(centroids[j,:], dataSet[i,:])
                if disJ < minDist:
                    minDist = disJ
                    minIndex = j
            if clusterAssent[i,0] != minIndex:
                clusterChanged = True
            clusterAssent[i,0] = minIndex
            minDist**2
        iterTime += 1
        sse = sum(clusterAssent[:,1])
        for cent in range(k):
            pstInClust = dataSet[nonzero(clusterAssent[:,0].A==cent)[0]]
            centroids[cent,:] = mean(pstInClust, axis=0)
        return centroids,clusterAssent

# 2维数据聚类效果显示
def datashow(dataSet, k, centroids, clusterAssment):  # 二维空间显示聚类结果
    num, dim = shape(dataSet)  # 样本数num ,维数dim
    if dim != 2:
        # print 'sorry,the dimension of your dataset is not 2!'
        return 1
    marksamples = ['or', 'ob', 'og', 'ok', '^r', '^b', '<g']  # 样本图形标记
    if k > len(marksamples):
        # print 'sorry,your k is too large,please add length of the marksample!'
        return 1
    # 绘所有样本
    for i in range(num):
        markindex = int(clusterAssment[i, 0])  # 矩阵形式转为int值, 簇序号
        # 特征维对应坐标轴x,y；样本图形标记及大小
        plt.plot(dataSet[i, 0], dataSet[i, 1], marksamples[markindex], markersize=6)
    # 绘中心点
    markcentroids = ['dr', 'db', 'dg', 'dk', '<r', 'sb', '<r']  # 聚类中心图形标记
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], markcentroids[i], markersize=15)
    plt.title('k-means cluster result')  # 标题
    plt.show()

if  __name__== '__main__':
    dataSet = loadData("/Users/scofield/MLRep/Data/K_MeansPlustestSet.txt")
    # print("dataSet",dataSet)
    dataMat = mat(dataSet)
    k = 4
    mm = randChosenCenter(dataMat,k)
    # print("mm",mm)
    mycentroids,myclusterAssent = kMeans(dataMat,k)
    # print("mycentroids",mycentroids)
    # print("myclusterAssent",myclusterAssent)
    datashow(dataMat, k, mycentroids, myclusterAssent)
    runTime = 3
    for i in range(runTime):
        plt.figure(i+1)
        mycentroids, myclusterAssent = kMeans(dataMat, k, kMeansPlusPlus)
        datashow(dataMat, k, mycentroids, myclusterAssent)
