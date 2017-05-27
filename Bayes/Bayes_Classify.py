# input a dataset
from numpy import *
from os import listdir

def loadDataSet():
    dataList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # the value of labels 0:positive 1:negative
    classLabels = [0,1,0,1,0,1]
    # return dataset and label list
    return dataList,classLabels

# create a distancted word list
def createWordList(dataSet):
    # create a null set used for distanct
    wordList = set()
    valist = []
    # loop the dataset
    for word in dataSet:
        # use the list store the word distancted
        # if word not in wordList:
        #     wordList.append(word)
        wordList = wordList|set(word)
    for ks in wordList:
        valist.append(ks)
    return valist


def createMatrixFromFile(wordList, inputData):
    # create a list with 0
    returnList = len(wordList)*[0]
    # loop each word in the input
    for word in inputData:
        # judge if the word in list
        if word in wordList:
            # bag Of words count the times of the words the in put have
            returnList[wordList.index(word)] += 1
            # Set of words each word one time
            # returnList[wordList.index(word)] = 1
    return returnList

# train the model get the prior posibality and posterior posibality
def trainBayes(dataList, classLabels):
    # count the samples
    countSamples = len(classLabels)
    # count the dimentions
    countdimens = len(dataList[0])
    # calculate the posibility of the sentence is negatice
    percemtng = sum(classLabels)/countSamples
    # store negative words
    cngsum = zeros(countdimens)
    # store positive words
    cposum = zeros(countdimens)
    # count the numbers of all the ngative words
    negcounts = 0
    # count the numbers of all the positive words
    poscounts = 0
    # loop each sample
    for i in range(countSamples):
        # if the sample is a negative sentence
        if classLabels[i] == 1:
            # store the times of negative word attained in the negative sentence
            cngsum += dataList[i]
            # store all the number of negative words
            negcounts += sum(dataList[i])
        else:
            # store the times of positive word attained in the negative sentence
            cposum += dataList[i]
            # store all the number of positive words
            poscounts += sum(dataList[i])
    perneg = cngsum/negcounts
    perpos = cposum/poscounts
    return percemtng,perneg,perpos

def trainBayes2(trainData, trainLabels):
    # count the samples
    countSamples = len(trainLabels)
    # count the dimentions of the data
    countDimens = len(trainData[0])
    # calculate the percentage of the negative samples
    negpercentage = sum(trainLabels)/countSamples
    # create a list with 1 used for store
    negwordnum = ones(countDimens)
    poswordnum = ones(countDimens)
    allnegnum = 2.0
    allposnum = 2.0
    for i in range(countSamples):
        if trainLabels[i] == 1:
            negwordnum += trainData[i]
            allnegnum += sum(trainData[i])
        else:
            poswordnum += trainData[i]
            allposnum += sum(trainData[i])
    pneg = log(negwordnum/allnegnum)
    ppos = log(poswordnum/allposnum)
    return negpercentage,pneg,ppos

# the parameters: verclassify artivle need be judgement percemtng percent of the negative centence in all centences
def classifyBayes(verclassify, percemtng, perneg, perpos):
    pn = sum(verclassify*perneg) + log(percemtng)
    pp = sum(verclassify*perpos) + log(1.0-percemtng)
    if pn>pp:
        return 1
    else:
        return 0

def spamTest():
    # store word list
    docList = []
    # store labels
    labelList = []
    for i in range(1,26):
        # read file and store the words in list
        wordList = cleanText(open('/Users/scofield/PycharmProjects/Cross/email/spam/%d.txt' % i,encoding='GBK').read())
        docList.append(wordList)
        labelList.append(1)
        # read file and store the words in list
        wordList = cleanText(open('/Users/scofield/PycharmProjects/Cross/email/ham/%d.txt' % i,encoding='GBK').read())
        docList.append(wordList)
        labelList.append(0)
    # create wordlist
    vocaList = createWordList(docList)
    trainSet=[]
    for k in range(0,50):
        trainSet.append(k)
    testSet = []
    # choose 10 sample as testset randomly the orher part as  trainset
    for i in range(10):
        randindex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randindex])
        trainSet.remove(trainSet[randindex])
    trainMatrix = []
    trainLabels = []
    # create trainMatrix
    for docindex in trainSet:
        trainMatrix.append(createMatrixFromFile(vocaList, docList[docindex]))
        trainLabels.append(labelList[docindex])
    # use trainMatrix exercise model
    ppam,perneg,ppos = trainBayes2(trainMatrix, trainLabels)
    errcount = 0
    for docindex in testSet:
        # create testMatrix
        doc = createMatrixFromFile(vocaList,docList[docindex])
        # judge if the predicted result is right the account the wrong number
        if classifyBayes(array(doc),ppam,perneg,ppos) != labelList[docindex]:
            errcount += 1
            print('classify error : ', docList[docindex])
    print("classify error percent", float(errcount/len(testSet)))
    return float(errcount/len(testSet))




def cleanText(bigString):
    import re
    # regular matching not a number and not a letter split the string
    listToken = re.split(r'\W*', bigString)
    # lower the string and loop the string[] choose the length of word large than 2
    return [tok.lower for tok in listToken if len(tok) > 2]

if __name__ == '__main__':

    # dataset, trainlabels = loadDataSet()
    # vocablist=createWordList(dataset)
    # print(vocablist)
    # testSample = ["my", "dog", "has", "flea", "problem", "help", "please", "my"]
    # print(testSample[0])
    # becreated = createMatrixFromFile(vocablist,testSample)
    # print(becreated)
    # traindata = []
    # for line in dataset:
    #     traindata.append(createMatrixFromFile(vocablist,line))
    # print(traindata)
    # negpercentage, pneg, ppos = trainBayes2(traindata, trainlabels)
    # print(negpercentage)
    # print(pneg)
    # print(ppos)
    # testSample=["stupid","dog","fool"]
    # testMatrix = createMatrixFromFile(vocablist, testSample)
    # print(testMatrix)
    # result = classifyBayes(testMatrix,negpercentage, pneg, ppos)
    # print(result)

    # ====================垃圾邮箱分类

    spamTest()
    # 多次运行取平均值
    # errorsum = 0.0
    # for i in range(10):
    #     # spamTest()无返回值时，就会报错：
    #     # TypeError: unsupported operand type(s) for +=: 'float' and 'NoneType'
    #     error = spamTest()
    #     errorsum += error
    # print('the average error:', errorsum / 10)
