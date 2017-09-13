from numpy import *
from sklearn import linear_model
def loadData():
    path = r"/Users/scofield/MLRep/Data/Delivery_Dummy.csv"
    data = genfromtxt(path, delimiter=',')
    return data

def getXY(data):
    x = data[1:,:-1]
    y = data[1:,-1]
    return x,y

if __name__ == '__main__':
    data = loadData()
    x,y = getXY(data)
    mlr = linear_model.LinearRegression()
    mlr.fit(x, y)
    x_test = [[90,2,0,0,1]]
    y_pred = mlr.predict(x_test)
    print("y_pred : ", y_pred)