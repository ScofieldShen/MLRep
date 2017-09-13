from numpy import *
from sklearn import linear_model
def loadData():
    dataPath = r"/Users/scofield/MLRep/Data/Delivery.csv"
    data = genfromtxt(dataPath, delimiter=',')
    return data

def getXY(data):
    x = data[:,:-1]
    y = data[:,-1]
    return x, y

if __name__ == '__main__':
    data = loadData()
    print("data : ", data)
    x,y = getXY(data)
    print("x", x.shape)
    print("y", y)
    lr = linear_model.LinearRegression()
    lr.fit(x, y)
    print("lr : ", lr)
    print("coefficients:", lr.coef_)
    print("intercept:", lr.intercept_)
    x_test = [[102, 6]]
    y_pred = lr.predict(x_test)
    print("predict : ", y_pred)
