# __*__coding:utf-8
from numpy import *
import matplotlib.pyplot as plt
def loadData(fileName):
    fr = open(fileName)
    strArr = [line.strip().split("\t") for line in fr.readlines()]
    print("strArr",strArr)
    dataSet = [list(map(float, str)) for str in strArr]
    print("dataSet",dataSet)
    return mat(dataSet)


def pca(dataSet, topNFea=9999999):
    meanVals = mean(dataSet, axis=0)
    meanRemov = dataSet - meanVals
    covMat = cov(meanRemov, rowvar=0)
    eigVals,eigVecs = linalg.eig(mat(covMat))
    eigValID = argsort(eigVals)
    eigValID = eigValID[:-(topNFea+1):-1]
    redEigVects = eigVecs[:,eigValID]
    lowDataMat = meanRemov * redEigVects
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat,reconMat

def replaceNanWithMEan():
    dataMat = loadData("/Users/scofield/MLRep/Data/pcasecom.data")
    numFea = shape(dataMat)[1]
    for i in range(numFea):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A)[0],i)])
        dataMat[nonzero(~isnan(dataMat[:, i].A)[0], i)] = meanVal
    return dataMat

def dataShow(dataMat, reconMat):
    fit = plt.figure()
    ax = fit.add_subplot(111)
    print("dataMat",dataMat)
    print("reconMat",reconMat)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], marker='^', s=80)
    ax.scatter(reconMat[:, 0], reconMat[:, 1], marker='o', s=20, c='red')
    plt.scatter([dataMat[:, 0]], [dataMat[:, 1]])
    plt.show()

if __name__ == '__main__':
    dataMat = loadData("/Users/scofield/MLRep/Data/pcatestSet.txt")
    topNFeature = 1
    dataSet,recMat = pca(dataMat, topNFeature)
    dataShow(dataMat, recMat)