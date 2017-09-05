from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
from numpy import *
import csv

def loaddata():
    data = open(r'/Users/scofield/MLRep/Data/AllElectronics.csv', 'r')
    # print("data : ", data)
    reader = csv.reader(data)
    # print(reader)
    header = next(reader)
    # print(header)
    featureList = []
    labelList = []
    for raw in reader:
        # print(raw)
        labelList.append(raw[len(raw)-1])
        rawDic = {}
        for i in range(1, len(raw)-1):
            # print(header[i])
            rawDic[header[i]] = raw[i]
        featureList.append(rawDic)
    return featureList,labelList
def createtree(featureList):
    dec = DictVectorizer()
    dummX = dec.fit_transform(featureList).toarray()
    # print(str(dummX))
    # print(dec.get_feature_names())
    lb = preprocessing.LabelBinarizer()
    dummY = lb.fit_transform(labelList)
    # print(str(dummY))
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clt = clf.fit(dummX, dummY)
    return clt,dec,dummX

if __name__ == '__main__':
    featureList,labelList = loaddata()
    clf,dec,dummX = createtree(featureList)
    with open("allElectronicInformationGainOri.dot", 'w') as f:
        tree.export_graphviz(clf, feature_names = dec.get_feature_names(), out_file = f)
        oneRow = dummX[0,:]
        # print("oneRow : ", oneRow)
    newRowX = oneRow
    newRowX[0] = 1
    newRowX[2] = 0
    print("newRow : ", newRowX)
    temp = newRowX.reshape(1,-1)
    predictY = clf.predict(temp)
    print("predictY : ", predictY)