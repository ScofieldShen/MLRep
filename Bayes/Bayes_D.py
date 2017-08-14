from numpy import *

def loadData():
    # 加载数据
    dataSet = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 加载标签数据
    classLabels = [0,1,0,1,0,1]
    return dataSet,classLabels

def createVocList(dataSet):
    # 创建一个set集合去重用
    wordList = set()
    # 循环dataSet中的每个元素
    for word in dataSet:
        # 将word元素转成set 与wordList做并集达到去重的效果
        wordList = wordList | set(word)
    return list(wordList)

def wordVecFromDoc(wordList, inputData):
    # 创建一个元素值为0的与wordList相同长度的列表
    returnrList = len(wordList)*[0]
    # 将输入数据的每个数据循环
    for word in inputData:
        # 如果字典中有word 就将word对应的wordList角标在returnrList中的元素设置为1
        if word in wordList:
            returnrList[wordList.index(word)] += 1
    return returnrList

def trainBayes(trainList, trainLabels):
    # 获取样本数
    numSamples = len(trainLabels)
    # 负面占比
    perrngcmt = sum(trainLabels)/numSamples
    # 字典长度
    countDimensation = len(trainList[0])
    # 创建一个用于存储 侮辱性言论单词出现次数的列表
    ngsum = zeros(countDimensation)
    # 创建一个用于存储 积极性言论单词出现次数的列表
    possum = zeros(countDimensation)
    # 侮辱性言论总单词数
    ngcount = 0
    # 积极性言论总单词数
    poscount = 0
    # 循环每个样本角标
    for i in range(numSamples):
        # 如果标签值是1
        if trainLabels[i] == 1:
            # 侮辱性言论的总单词数
            ngcount += sum(trainList[i])
            # 每个单词在侮辱性言论中分别出现的次数
            ngsum += trainList[i]
        else:
            poscount += sum(trainList[i])
            possum += trainList[i]
    perng = ngsum/ngcount
    perpos = possum/poscount
    return perrngcmt,perng,perpos

def trainBayes1(trainList, trainLabels):
    # 获取样本数
    numSamples = len(trainLabels)
    # 负面占比
    perrngcmt = sum(trainLabels)/numSamples
    # 字典长度
    countDimensation = len(trainList[0])
    # 创建一个用于存储 侮辱性言论单词出现次数的列表
    ngsum = ones(countDimensation)
    # 创建一个用于存储 积极性言论单词出现次数的列表
    possum = ones(countDimensation)
    # 侮辱性言论总单词数
    ngcount = 2.0
    # 积极性言论总单词数
    poscount = 2.0
    # 循环每个样本角标
    for i in range(numSamples):
        # 如果标签值是1
        if trainLabels[i] == 1:
            # 侮辱性言论的总单词数
            ngcount += sum(trainList[i])
            # 每个单词在侮辱性言论中分别出现的次数
            ngsum += trainList[i]
        else:
            poscount += sum(trainList[i])
            possum += trainList[i]
    perng = ngsum/ngcount
    perpos = possum/poscount
    return perrngcmt,perng,perpos

def classifyBayes(vecClassify, perrngcmt,perng,perpos):
    ng = sum(vecClassify*perng) + log(perrngcmt)
    pos = sum(vecClassify*perng) + log(1.0 - perrngcmt)
    if ng > pos:
        return 1
    else:
        return 0

def cleanTest(bigString):
    import re
    token = re.split(r'\W*', bigString)
    return [tok.lower for tok in token if len(tok) > 2]

def testSpamst():
    wordList = []
    labelList = []
    for i in range(1, 26):
        wordLine = cleanTest(open('/Users/scofield/MLRep/Data/email/ham/%d.txt' % i, encoding='GBK').read())
        wordList.append(wordLine)
        labelList.append(0)
        wordLine = cleanTest(open('/Users/scofield/MLRep/Data/email/spam/%d.txt' % i, encoding='GBK').read())
        wordList.append(wordLine)
        labelList.append(1)
    vocList = createVocList(wordList)
    trainSet = []
    for k in range(50):
        trainSet.append(k)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        # trainSet.remove(randIndex)
        delete(trainSet, randIndex)
    trainMatrix = []
    trainLabels = []
    print("testSet : ",testSet)
    print("trainSet : ",trainSet)
    for index in trainSet:
        trainMatrix.append(wordVecFromDoc(vocList, wordList[index]))
        trainLabels.append(labelList[index])
    perrngcmt, perng, perpos = trainBayes1(array(trainMatrix),array(trainLabels))
    errorCount = 0
    for index in testSet:
        wordvor = wordVecFromDoc(vocList, wordList[index])
        if classifyBayes(wordvor, perrngcmt, perng, perpos) != labelList[index]:
            errorCount += 1
    print("errorCount : ", errorCount)

    return float(errorCount)/len(testSet)

if __name__ == '__main__':
    # dataSet, classLabels = loadData()
    # wordList = createVocList(dataSet)
    # print("wordList : ",wordList)
    # trainList = []
    # for li in dataSet:
    #     trainList.append(wordVecFromDoc(wordList, li))
    # print("trainList ： ",trainList)
    # percentng, perpos, perng = trainBayes1(trainList, classLabels)
    # print("end : ",percentng,perpos,perng)
    # testList= ['garbage', 'stupied']
    # wordvec = wordVecFromDoc(wordList, testList)
    # predictres = classifyBayes(wordvec, percentng, perpos, perng)
    # print("predictres",predictres)
    testSpamst()