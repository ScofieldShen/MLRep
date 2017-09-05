# __*__ coding : utf-8
from KNN.KNN_Classify import *


def classify_person():
    resultList = ['does not like','small like','large like']
    fmiles = float(eval(input("this is flymiles")))
    gametime = float(eval(input("the time of play games")))
    icecream = float(eval(input("the used icecream")))
    dataMatrix,labels = file2Matrix("/Users/scofield/PycharmProjects/Cross/dataknn.txt")
    print(dataMatrix)
    print(labels)
    normalMatrix,ranges,minSample = normalData(dataMatrix)
    inarr = array([fmiles,gametime,icecream])
    result = knn_classify((inarr-minSample)/ranges,normalMatrix,labels,3)
    print("this is the result %d" % labels[result[-1]])


if __name__ == '__main__':
    classify_person()