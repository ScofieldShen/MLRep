import numpy as np
from sklearn.datasets import load_digits
from DL.NeuralNW.NeuralNetWork import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

def getDigits():
    digits = load_digits()
    x = digits.data
    y = digits.target
    x -= x.min()
    x /= x.max()
    return x,y

if __name__ == '__main__':
    x, y = getDigits()
    # print(x)
    # print(y)
    nn = NeuralNetwork([64,100,10], 'logistic')
    x_train,x_test,y_train,y_test = train_test_split(x, y)
    label_train = LabelBinarizer().fit_transform(y_train)
    label_test = LabelBinarizer().fit_transform(y_test)
    # print(x_train[0])
    # print(label_train[0])
    nn.fit(x_train, label_train, epochs=3000)
    predictions = []
    for i in range(x_test.shape[0]):
        o = nn.predict(x_test[i])
        predictions.append(np.argmax(o))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))