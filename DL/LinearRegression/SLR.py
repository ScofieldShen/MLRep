import numpy as np

def fitLR(x, y):
    n = len(x)
    sumup = 0
    sumdown = 0
    for i in range(0, n):
        sumup += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        sumdown += (x[i] - np.mean(x))**2
    k = sumup/float(sumdown)
    b = np.mean(y) - k*np.mean(x)
    return k,b

def predict(x, k , b):
    return k*x + b

if __name__ == '__main__':
    x = [1, 3, 2, 1, 3]
    y = [14, 24, 18, 17, 27]
    k,b = fitLR(x, y)
    print("k", k)
    print("b", b)
    y_pred = predict(6, k, b)
    print(y_pred)