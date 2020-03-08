import numpy as np
import matplotlib.pyplot as plt

X = np.array([[3, 3, 1], [4, 3, 1], [1, 1, -1]])


def set(w1=0, w2=0, b=0, rate=1):
    return w1, w2, b, rate


def dell():
    w1, w2, b, rate = set()
    xunhuan = 0
    sum = 0
    while (xunhuan < 3):
        xunhuan = 0
        for i in range(len(X)):
            Y = w1 * X[i][0] + w2 * X[i][1] + b
            if (Y * X[i][2] <= 0):  # 未能正确分类  更新
                xunhuan = 0
                w1 = w1 + rate * X[i][2] * X[i][0]
                w2 = w2 + rate * X[i][2] * X[i][1]
                b = b + rate * X[i][2]
                sum += 1
                print("{0}+is  : {1}x +{2}y +{3}".format(sum, w1, w2, b))  # 正则输出
            else:
                xunhuan += 1
    return 0


# 面向对象
class MyPerceptron:
    def __init__(self):
        self.w = None
        self.b = 0
        self.learn = 1

    def fit(self, X, Y):
        # 用样本点特征值更新初始w  比如 x1=(3,3)  初始w=(0,0)
        self.w = np.zeros(X.shape[1])  # 看给的训练集X中一个样本的维度
        i = 0
        while i < X.shape[0]:  # 去出来每个样本
            x = X[i]
            y = Y[i]
            # 如果是误分类点 更新w,b  注意是随机梯度下降法 找到误分类点，立刻更新
            if y * (np.dot(self.w, x) + self.b) <= 0:
                self.w = self.w + self.learn * np.dot(y, x)
                self.b = self.b + self.learn * y
                i = 0  # 如果是误分类点，更新后要从头检验
            else:
                i += 1
        return self.w, self.b


if __name__ == '__main__':
    dell()
    print()
    X = np.array([[3, 3], [4, 3], [1, 1]])
    Y = np.array([1, 1, -1])
    perceptron = MyPerceptron()
    w, b = perceptron.fit(X, Y)
    print(w, "   ", b)
