import numpy as np
import random

def generateData(numPoints, bias, variable):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=(numPoints))
    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) + variable
    return x,y

def gradientDescent(x, y, theta, alpha, m, numIterations):
    XTran = np.transpose(x)
    for i in range(numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss**2)/2*m
        gradient = np.dot(XTran, loss)/m
        theta = theta - alpha*gradient
    return theta

if __name__ == '__main__':
    x,y = generateData(100, 25, 10)
    # print("x : ", x)
    # print("y : ", y)
    m,n = np.shape(x)
    numIterations = 100000
    alpha = 0.0005
    theta = np.ones(n)
    theta = gradientDescent(x, y, theta, alpha, m, numIterations)
    print("theta : ", theta)