import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils

# 超参数
learning_rate = 0.075
iterations = 2500
# 构建一个L层的神经网络，包含L-1个隐藏层和1个输出层


class deep_net():
    def __init__(self, layers_dims):
        self.parameters = {}  # 构建一个字典来存储权重
        self.grads = {}  # 构建一个字典来存储梯度
        L = len(layers_dims)-1
        for l in range(1, L+1):  # 从第一层开始，第0层是输入层
            self.parameters["W" + str(l)] = np.random.randn(layers_dims[l],
                                                            layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
            self.parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        assert(self.parameters["W" + str(l)].shape ==
               (layers_dims[l], layers_dims[l-1]))
        assert(self.parameters["b" + str(l)].shape == (layers_dims[l], 1))

    def linear_forward(self, A_prev, W, b):  # 每一层的线性部分
        # A是上一层的激活值
        Z = np.dot(W, A_prev) + b
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        cache = (A_prev, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):  # 每一层的线性+激活
        """
        A_prev 是来自上一层的激活值，这一层的输入
        W，b是本层的权重，偏差
        """
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
        assert(A.shape == (W.shape[0], A_prev.shape[1]))
        # linear_cache装有(上层的A也即本层的输入，本层的W，本层的b)，activation_cache就是本层的Z
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward(self, X, activation):  # 总共L层的前向传播
        """
        caches中包含每一层的（上一层的A即本层的输入，本层的W，本层的b，本层的Z）
        """
        caches = []  # 前向传播中L层的所有缓存
        A = X  # 第一层的激活值就是输入向量
        L = len(self.parameters) // 2
        for l in range(1, L):  # 1到L-1层
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation)
            caches.append(cache)
        # 第L层是输出层，使用sigmoid
        AL, cache = self.linear_activation_forward(
            A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], "sigmoid")
        caches.append(cache)
        assert(AL.shape == (1, X.shape[1]))
        return AL, caches

    def loss(self, AL, Y):
        m = Y.shape[1]
        cost = -1/m*np.sum(np.multiply(np.log(AL), Y) +
                           np.multiply(np.log(1-AL), 1-Y))
        cost = np.squeeze(cost)
        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)  # 这是为了计算上一层反向传播的dZ
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        # 先激活函数求导算出dZ
        linear_cache, activation_cache = cache  # 取出两部分缓存
        if activation == 'relu':
            dZ = relu_backward(dA, activation_cache)  # 求导需要dA和Z
        elif activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
        # 随后对参数线性求导
        dA_prev, dW, db = self.linear_backward(
            dZ, linear_cache)  # dA_prev会用作上一层的激活函数求导
        return dA_prev, dW, db

    def backward(self, AL, Y, caches, activation):
        assert(AL.shape == Y.shape)
        L = len(caches)
        m = AL.shape[1]
        # 先对第L层的sigmoid求导再线性求导(也可以调用linear_activation_backward函数)
        dZL = AL-Y
        current_cache = caches[L-1]
        linear_cache, activation_cache = current_cache
        A_prev, WL, bL = linear_cache
        dWL = (1 / m) * np.dot(dZL, A_prev.T)
        dbL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
        dA_prev = np.dot(WL.T, dZL)  # 这是为了计算上一层反向传播的dZ
        self.grads['dW' + str(L)], self.grads['db' + str(L)] = dWL, dbL
        # 对隐层反向传播
        dA_prev_temp = dA_prev
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, self.grads['dW' + str(l + 1)], self.grads['db' + str(
                l + 1)] = self.linear_activation_backward(dA_prev_temp, current_cache, activation)

    def update(self):
        L = len(self.parameters) // 2  # 整除
        for l in range(1, L+1):
            self.parameters["W" + str(l)] = self.parameters["W" +
                                                            str(l)] - learning_rate * self.grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" +
                                                            str(l)] - learning_rate * self.grads["db" + str(l)]

    def predict(self, X, activation):
        AL, caches=self.forward(X, activation)
        return AL


# 识别猫
def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
    # 平坦+归一化
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    # 构建神经网络
    layers_dims = [12288, 64, 32, 4, 1]
    mynet = deep_net(layers_dims)
    # 训练
    costs = []
    for i in range(iterations):
        AL, caches = mynet.forward(train_x, 'relu')
        cost = mynet.loss(AL, train_y)
        costs.append(cost)
        mynet.backward(AL, train_y, caches, 'relu')
        mynet.update()
        if i % 100 == 0:
            print('第' ,i ,'轮的cost为:', cost)

    predictions = mynet.predict(test_x,'relu')
    print('神经网络的准确性: %d' % float((np.dot(test_y, predictions.T) +
          np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100) + '%')

    plt.plot(costs)
    plt.show()


if __name__ == "__main__":
    main()
