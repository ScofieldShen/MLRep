# __*__ coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

# read data
def loadData(fileName):
    # open file
    fr = open(fileName)
    # create list
    dataSet = []
    # read each line
    for line in fr.readlines():
        # kick blank and splited with "\t"
        lineStr = line.strip().split("\t")
        # change str to float
        curLine = list(map(float, lineStr))
        # add each line in dataSet
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

# draw original dataset
def originalDataShow(dataSet):
    # the number of samples and dimentions
    samples,dim = shape(dataSet)
    # sample picture targets
    markeSamples = ["ob"]
    for i in range(samples):
        plt.plot(dataMat[i,0], dataMat[i,1], markeSamples[0], markersize=5)
    #     title
    plt.title("original dataSet")
    plt.show()

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

def kMeansSSE(dataSet, k, calMens = calDistan, createCen = randChosenCenter):
    m = shape(dataSet)[0]
    clusterAssent = mat(zeros((m,2)))
    centroids = createCen(dataSet, k)
    sseOld = 0
    sseNew = inf
    iterTimes = 0
    while(abs(sseNew-sseOld) > 0.001):
        sseOld = sseNew
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                disJ = calMens(centroids[j,:], dataSet[i,:])
                if disJ < minDist:
                    minDist = disJ
            clusterAssent[i,:] = minIndex, minDist**2
        iterTimes += 1
        sseNew = sum(clusterAssent[:,1])
        for cent in range(k):
            ptsInCluster = dataSet[nonzero(clusterAssent[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInCluster, axis=0)
    return centroids,clusterAssent


# 2维数据聚类效果显示
def datashow(dataSet, k, centroids, clusterAssment):  # 二维空间显示聚类结果
    from matplotlib import pyplot as plt
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

if __name__ == '__main__':
    # load data
    dataSet = loadData("/Users/scofield/MLRep/Data/K_MeanstestSet.txt");
    dataMat = mat(dataSet)
    # print("dataMat",dataMat)
    # show original data
    # originalDataShow(dataMat)
    # Kmeans
    k = 4
    # myCentRoids,clusterAssent = kMeans(dataMat,k)
    myCentRoids,clusterAssent = kMeansSSE(dataMat,k)
    datashow(dataMat, k, myCentRoids, clusterAssent)
