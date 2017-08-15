def loadDLFromOneFile(fileName):
    # 统计维度数
    numDimensations = len(open(fileName).readline().strip().split("\t")) - 1
    # 读取所有数据
    lines = open(fileName).readlines()
    # 存储维度数据
    dataSet = []
    # 存储标签
    classLabels = []
    # 循环每行数据
    for line in lines:
        dataList = []
        # 将每行数据去重切分
        strData = line.strip().split("\t")
        for i in range(numDimensations):
            dataList.append(float(strData[i]))
        dataSet.append(dataList)
        # 获取每行的最后一列存储到标签数据
        classLabels.append(float(strData[-1]))
    return dataSet,classLabels

if __name__ == '__main__':
    loadDLFromOneFile("")