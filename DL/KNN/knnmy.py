import math
import csv
import random
import operator

# 加载数据
def loadData(fileName, splitPer, trainSet, testSet):
    # 打开文件转为csv文件
    with open(fileName, 'r') as csvfile:
        # 读取csv文件数据
        lines = csv.reader(csvfile)
        # 将数据转换为list
        dataSet = list(lines)
        # 循环每行数据
        for x in range(len(dataSet) - 1):
            # 将每个特征数据转为float类型
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
            # 将数据拆分为训练集和测试集
            if random.random() < splitPer:
                trainSet.append(dataSet[x])
            else:
                testSet.append(dataSet[x])
    return trainSet,testSet

# 计算距离
def uclideanDistance(inst1, inst2, length):
    distance = 0
    for x in range(length):
        # 求差的平方和
        distance += math.pow(inst1[x]-inst2[x], 2)
    # 返回距离平方和
    return math.sqrt(distance)

# 统计前k个neighbor
def getNeighbors(trainSet, testInst, k):
    distances = []
    # 特征维度数
    length = len(testInst)-1
    # 计算测试样本与每个训练样本的距离
    for x in range(len(trainSet)):
        dist = uclideanDistance(testInst, trainSet[x], length)
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1), reverse=True)
    print("sortedVotes : ",sortedVotes)
    return sortedVotes[0][0]

def getAccuracy(testSet, preDictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == preDictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    trainSet = []
    testSet = []
    splitPer = 0.7
    trainSet,testSet = loadData(r'/Users/scofield/MLRep/Data/irisdata.txt', splitPer, trainSet, testSet)
    print("trainSet : ",trainSet)
    print("testSet : ", testSet)
    k = 5
    predictions = []
    correct = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        print(neighbors)
        result = getResponse(neighbors)
        print("result : ",result)
        # predictions.append(result)
        # print("predictions : ", repr(predictions))
        print("testSet : ", testSet[x][-1])
    #     if result == testSet[x][-1]:
    #         correct.append(x)
    # print("correct : ", len(correct))
    # accuracy = getAccuracy(testSet, predictions)
    # print("accuracy : ", repr(accuracy) + %)

if __name__ == '__main__':
    main()