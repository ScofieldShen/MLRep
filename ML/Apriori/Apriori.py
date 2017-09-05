# coding:utf-8
from numpy import *
import itertools
def loadData():
    dataSet = [[1, 2, 5],  # 1
               [2, 4],
               [2, 3],
               [1, 2, 4],  # 4
               [1, 3],
               [2, 3],
               [1, 3],
               [1, 2, 3, 5],  # 8
               [1, 2, 3]]
    return dataSet

def loadDataSet1(): #说明生成的关联规则
    dataSet= [[1, 3, 4],
	        [2, 3, 5],
			[1, 2, 3, 5],
            [1, 2, 3, 4,5],
			[2, 5]]
    return  dataSet


def loadDataSet2(): #《机器学习实战》中的数据
    dataSet=[[1,3,4],
             [2,3,5],
             [1,2,3,5],
             [2,5]]
    return dataSet

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

# create all frequent item set
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    # print("C1 : ", C1)
    D = list(map(set, dataSet))
    L1,supportData = CKToLK(D, C1, minSupport)
    L = [L1]
    k = 2
    # print("L", L[0])
    while(len(L[k-2]) > 0):
        CK = candidateGen(L[k-2], k-1)
        # print("CK", CK)
        LK,superK = CKToLK(D, CK, minSupport)
        # print("LK : ",LK)
        supportData.update(superK)
        # print("superK : ", superK)
        L.append(LK)
        k += 1
    return L,supportData

def CKToLK(D, CK, minSupport):
    CKCount = {}
    # print("D",D)
    # print("CK",CK)
    for transaction in D:
        for ckItem in CK:
            if ckItem.issubset(transaction):
                if ckItem not in CKCount.keys():
                    CKCount[ckItem] = 1
                else:
                    CKCount[ckItem] += 1
    numTransactions = float(len(D))
    retList = []
    supportData = {}
    for key in CKCount:
        support = CKCount[key]/numTransactions
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    return retList,supportData

def candidateGen(LK, k):
    candidateProun = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(LK[i])[:k-1]
            L2 = list(LK[j])[:k-1]
            L1.sort()
            L2.sort()
            if L1 == L2:
                candidateC = LK[i]|LK[j]
                addFlag = True
                for iter in itertools.combinations(candidateC, k):
                    subcandidate = frozenset(list(iter))
                    if subcandidate not in LK:
                        addFlag = False
                        break
                if addFlag == True:
                    candidateProun.append(candidateC)
    return candidateProun

# generate association roles L2,L3产生
# imput: frequent items set support set believe set
# output: follow the least believe set
def generateResult(L, supportData, minConf):
    strongResultList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConsequent(freqSet, H1, supportData, strongResultList, minConf)
            else:
                calConf(freqSet, H1, supportData, strongResultList, minConf)
    return strongResultList



def rulesFromConsequent(freqSet, H, supportData, ruleList, minConf):
    m = len(H[0])
    if m ==1:
        H = calConf(freqSet, H, supportData, ruleList, minConf)
    if(len(freqSet) > (m + 1)):
        Hmp1 = candidateGen(H, m)
        Hmp1 = calConf(freqSet, Hmp1, supportData, ruleList, minConf)
        if(len(Hmp1) > 1):
            rulesFromConsequent(freqSet, Hmp1, supportData, ruleList, minConf)

def calConf(freqSet, H, supportData, ruleList, minConf):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf > minConf:
            prunedH.append(conseq)
            ruleList.append((freqSet-conseq, conseq, conf))
    return prunedH



if __name__ == '__main__':
    # dataSet = loadData()
    # print("data1 : ", dataSet)
    # L,supportData = apriori(dataSet, minSupport=0.2)
    # print("L : ", L)
    # print("supportData : ", supportData)

    # rules= generateResult(L, supportData, minConf=0.6)
    # print("rules", rules)
    dataSet = []
    for line in open("/Users/scofield/MLRep/Data/apriorimushroom.data").readlines():
        strArr = line.strip().split()
        set1 = []
        for str in strArr:
            set1.append(int(str))
        dataSet.append(set1)
    # print("dataSet", dataSet)
    L, supportData = apriori(dataSet, minSupport=0.3)
    print("L", L[1])
    # for item in L[1]:
    #     if item.intersection[2]:
    #         print("L2 Set : ", item)
    # for item in L[3]:
    #     if item.intersection[2]:
    #         print("L4 Set : ", item)
