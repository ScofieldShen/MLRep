import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

if __name__ == '__main__':
    a = tanh_deriv(8)
    print(logistic_derivative(a))

class NeuralNetwork:
    # init 相当于构造函数 self相当于this layers列表存储每层的单元数
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: 一个列表包含每层的单元数，至少两个值
        :param activation: 使用的派生函数
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        #     初始化权重
        self.weight = []
        for i in range(1, len(layers) - 1):
            self.weight.append((2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1)*0.25)
            self.weight.append((2*np.random.random((layers[i] + 1, layers[i + 1])) - 1)*0.25)

    def fit(self, x, y, learning_rate=0.2, epochs=10000):
        # 至少是两层
        x = np.atleast_2d(x)
        # 创建一个元素值为1的 矩阵
        temp = np.ones([x.shape[0], x.shape[1]+1])
        # 把x的所有的值赋给temp的每个实例的第一到倒数第二行
        temp[:, 0:-1] = x
        x = temp
        y = np.array(y)
        # 迭代预设的循环次数
        for k in range(epochs):
            # 随机抽取一行
            i= np.random.randint(x.shape[0])
            alpha = [x[i]]
            # 正向更新 计算单元的值 alpha包含所有神经元
            for l in range(len(self.weight)):
                alpha.append(self.activation(np.dot(alpha[l], self.weight[l])))
            # 计算误差
            error = y[i] - alpha[-1]
            deltas = [error * self.activation_deriv(alpha[-1])]
            # 反向更新
            for l in range(len(alpha) -2, 0, -1):
                deltas.append(deltas[-1].dot(self.weight[l].T) * self.activation_deriv(alpha[l]))
            deltas.reverse()
            for i in range(len(self.weight)):
                layer = np.atleast_2d(alpha[i])
                delta = np.atleast_2d(deltas[i])
                self.weight[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        alpha = temp
        for l in range(0, len(self.weight)):
            # print(alpha)
            # print(self.weight[l])
            alpha = self.activation(np.dot(alpha, self.weight[l]))
        return alpha
