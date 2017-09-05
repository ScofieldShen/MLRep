# coding: utf-8
from numpy import *
def loadSimpData():
    simpData = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpData


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def createTree(dataSet, minSup=1):
    headTable = {}
    for trans in dataSet:
        for item in trans:
            headTable[item] = headTable.get(item, 0) + dataSet[trans]
    for key in headTable.keys():
        if headTable[key] < minSup:
            delete(headTable, key)
    freItemSet = set(headTable.keys())
    if len(freItemSet) == 0:
        return None,None
    for k in headTable:
        headTable[k] = [headTable[k], None]
    for tranSet,count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freItemSet:
                localD[item] = headTable[item][0]
        if len(localD) > 0:
            orderItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        # 结点名称
        self.name = nameValue
        # 结点出现次数
        self.count = numOccur
        # 链接相似元素项（横向找）
        self.nodeLink = None
        # 父结点（由上往下找）
        self.parent = parentNode  # needs to be updated
        # 子结点（由下往上找）
        self.children = {}
        # 增加次数

    def inc(self, numOccur):
        self.count += numOccur
        # 显示FP树结构
        # 当前结点显示的最前面有ind个空格，其子结点输出空格数ind+1

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def updateTree(items, inTree, headTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode[items[0], count, inTree]
        if headTable[items[0]][1] == None:
            headTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headTable, count)


# 从头指针的nodeToTest开始
# 沿着nodeLink直到链表末尾
# 然后将链表末尾指向新结点targetNode
# 确保结点链接指向树中该元素项的每一个实例
def updateHeader(nodeToTest, targetNode):  # this version does not use recursion
    while (nodeToTest.nodeLink != None):  # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


if __name__ == '__main__':
    minSup = 3
    simpData = loadSimpData()
    initSet = createInitSet(simpData)
    print("initSet", initSet)
    myFPtree,myHeadTable = createTree(simpData, minSup)
    myFPtree.disp()
