# __*__ coding: utf-8
from numpy import *
import matplotlib.pyplot as plt

def createTree(dataSet, featureNames):
    labelList = [sample[-1] for sample in dataSet]
    # print('labelList',labelList)
    # the end conditions first : the dataSet belongs to the same side return the sign of the category
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]
    # the end conditions second : when all the features used up use the largest number of signs
    if len(labelList) == 1 :
        return majorityCount(labelList)
    bestFeature = chooseBestMessageGain(dataSet)
    bestFeatureNames = featureNames[bestFeature]
    # print('bestFeatureNames : ', bestFeatureNames)
    myTree = {bestFeatureNames:{}}
    delete(featureNames,bestFeature)

    featureValues = [example[bestFeature] for example in dataSet]
    distinctValues = set(featureValues)

    for value in distinctValues:
        subFeatureNames = featureNames[:]
        subDataSet = splitDataSet(dataSet,bestFeature,value)
        subTree = createTree(subDataSet,subFeatureNames)
        myTree[bestFeatureNames][value] = subTree
    return myTree

def choseBestMessageGain(dataSet):
    numDimensations = len(dataSet[0]) - 1
    bestEntropy = calculateEntropy(dataSet)
    allConditionEntropy = []
    allSplitInfo = []
    for i in range(numDimensations):
        featureList = (example[i] for example in dataSet)
        [splitInfo, distinctFeatureValues] = calculateSplitInfo(featureList)
        allSplitInfo.append(splitInfo)
        conditionEntropy = 0.0
        for value in distinctFeatureValues:
            subDataSet = splitDataSet(dataSet, i, value)
            subEntropy = calculateEntropy(subDataSet)
            conditionEntropy += float(len(subDataSet)) / float(len(dataSet))*subEntropy
        allConditionEntropy.append(conditionEntropy)
        infoGain = bestEntropy*ones(numDimensations) - array(allConditionEntropy)
        infoGainRatio = infoGain/array(splitInfo)
        bestFeatureIndex =argsort(-infoGainRatio)[0]
    return bestFeatureIndex


# choose the best classify of dataset then choose the BestMessageGain ID3 tree
def chooseBestMessageGain(dataSet):
    # calculate the dimensations of dataset
    numDimensations = len(dataSet[0])-1
    # print("numDimensations", numDimensations)
    # calculate the whole entropy
    baseEntropy = calculateEntropy(dataSet)
    # init the best gain o
    bestInfoGain = 0.0
    # init the number of the best features rank as -1
    bestFeature = -1
    # loop all the dimennsations of feature and get the best entropy
    # print('numDimensations : ',numDimensations)
    for i in range(numDimensations):
        featureList = (feature[i] for feature in dataSet)
        # distinct the featureList
        distFeatureList = set(featureList)
        newEntropy = 0.0
        for value in distFeatureList:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calculateEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def calculateSplitInfo(featureList):
    numSamples = len(featureList)
    distinctFeatureValues = list(set(featureList))
    numValues = [featureList.count(feature) for feature in featureList]
    featureRateList = [float(fea)/numSamples for fea in numSamples]
    logList = [item*math.log(item,2) for item in featureRateList]
    splitInfo = -sum(logList)
    return splitInfo,distinctFeatureValues

# classify dataSet depend on the given feature
def splitDataSet(dataSet,i,value):
    # define an array
    retDataSet = []
    # loop the dataSet
    # print("dataSet", dataSet)
    for vec in dataSet:
        # print("vec[i]", vec[i])
        # judge the value of
        if vec[i] == value:
            # get elements from 0 to i
            reduceFeaVec = vec[:i]
            # print("reduceFeaVec",reduceFeaVec)
            # add each elements of the list which from i+1 to the end to array
            reduceFeaVec.extend(vec[i+1:])
            # add an element to retDataSet
            retDataSet.append(reduceFeaVec)
            # print("retDataSet",retDataSet)
    return retDataSet

# input dataset calculate Entropy
def calculateEntropy(dataSet):
    # print("len(dataSet)",len(dataSet))
    # count samples
    numSamples = len(dataSet)
    # split the labels and feature
    labelList = [data[-1] for data in dataSet]
    # print("labelList",labelList)
    # classify labels and count
    countLabel = dict([(i,labelList.count(i)) for i in labelList])
    # print("countLabel", countLabel)
    # calculate Entropy
    entropy = 0.0
    # loop each pair in countLabel
    for key in countLabel:
        # calculate each classified label's percentage
        prob = float(countLabel[key])/numSamples
        entropy -= prob * math.log(prob,2)
    # print("Entrpy", Entrpy)
    return entropy

def majorityCount(labelList):
    items = dict([(labelList.count(i), i) for i in labelList])
    return items[max(items.keys())]


def loadDataSet(filename):
    dataList = []
    fr = open(filename)
    for line in fr.readlines():
        fields = line.strip().split("\t")
        dataList.append(fields)
    return dataList

def file2Matrix(filename):
    lines = open(filename).readlines()
    numSamples = len(lines)
    dataMatrix = zeros(numSamples, 3)
    classLabel = []
    index = 0
    for line in lines:
        fields = line.strip().split("\t")
        dataMatrix[index, :] = fields[0:3]
        classLabel.append(int(fields[-1]))
        index += 1
    return dataMatrix,classLabel

def createDataSet():
    dataSet = [[1,0,0,1,'no'],
               [1,0,0,2,'no'],
               [1,1,0,2,'yes'],
               [1,1,1,1,'yes'],
               [1,0,0,1,'no'],
               [2,0,0,1,'no'],
               [2,0,0,2,'no'],
               [2,1,1,2,'yes'],
               [2,0,1,3,'yes'],
               [2,0,1,3,'yes'],
               [3,0,1,3,'yes'],
               [3,0,1,2,'yes'],
               [3,1,0,2,'yes'],
               [3,1,0,3,'yes'],
               [3,0,0,1,'no']]
    featureNames = ['age','work','house','credibility']
    return dataSet,featureNames

# start draw the constructed tree

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    sortedKeys = sorted(myTree.keys())
    firstStr = sortedKeys[0]
    seconDict = myTree[firstStr]
    for key in seconDict.keys():
        # if the node is dicctionaries otherwise it is leaf node
        if type(seconDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(seconDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    sortedKeys = sorted(myTree.keys())
    firstStr = sortedKeys[0]
    seconDict = myTree[firstStr]
    for key in seconDict.keys():
        if type(seconDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(seconDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


def plotMidText(entrPt, parentPt, txtString):
    xMid = (parentPt[0] - entrPt[0])/2.0 + entrPt[0]
    yMid = (parentPt[1] - entrPt[1])/2.0 + entrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va="center",ha="center",rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depths = getTreeDepth(myTree)
    sortedKeys = sorted(myTree.keys())
    firstStr = sortedKeys[0]
    centerPt = (plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff)
    plotMidText(centerPt, parentPt, nodeTxt)
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0/plotTree.totalD
    # keys = secondDict.keys()
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], centerPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.x0ff, plotTree.y0ff), centerPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), centerPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0/plotTree.totalD

def createPlot(tree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.x0ff = -0.5/plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(tree, (0.5,1.0), '')
    plt.show()


if __name__ == '__main__':
    # featNamesOri = ['age', 'revenue', 'student', 'credit']
    # myData = loadDataSet("/Users/scofield/MLRep/Data/C45dataset.dat")
    # loan data
    # myData,featureNames = createDataSet()
    # featureNames = featNamesOri[:]
    # myTree = createTree(myData,featureNames)
    # print(myTree)
    # createPlot(myTree)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    mydata = loadDataSet("/Users/scofield/MLRep/Data/C45lenses.txt")
    myTree= createTree(mydata, lensesLabels)
    print(myTree)
    createPlot(myTree)