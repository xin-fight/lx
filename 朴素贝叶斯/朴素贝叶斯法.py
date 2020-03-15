import numpy as np
import pandas as pd


class NaiveBayes():
    def __init__(self, lambda_):
        self.lambda_ = lambda_  # 贝叶斯系数 为0时是极大似然估计
        self.y_types_count = None  # y的类型  数量
        self.y_types_proba = None  # y的类型  概率
        self.x_types_proba = dict()  # 条件概率： xi的编号 xi的取值 y的类型 概率  要构造一个字典

    def fit(self, X_train, y_train):
        self.y_types = np.unique(y_train)  # y 的全部取值种类
        X = pd.DataFrame(X_train)
        y = pd.DataFrame(y_train)
        self.y_types_count = y[0].value_counts()  # 对y的各个取值的（类型） 数量 统计
        # y的各个取值的（类型） 概率 计算
        self.y_types_proba = (self.y_types_count + self.lambda_) / (y.shape[0] + len(self.y_types) * self.lambda_)  # y.shape[0]代表有多少个样本 self.y_types代表y全部的取值种类

        # 条件概率： xi的编号 xi的取值 y的类型 概率
        # 计算对于某个y取值的条件下  x某个特征的某个取值  的概率

        # 思路： 对每个特征中的值（例如：x1：1，2，3） 找到每个y取值（1，-1） 对应的个数
        '''
        要结合公式去理解
        '''
        for idx in X.columns:  # 遍历x列数 得到X特征值的列序  一列对于一个特征值
            for j in self.y_types:  # 遍历y的取值的种类
                # 对于y的某个取值j  (y==j).values返回所有满足的索引值   即返回y中与j相同的索引位置
                p_x_y = X[(y == j).values][idx].value_counts()  # 求y==i为真的数据点的第idx的特征值的个数

                print("y = ", j)
                print(X[(y == j).values][idx].value_counts())
                print("*" * 10)

                print(p_x_y.index)
                for i in p_x_y.index:  # 计算条件概率   p_x_y.index是保存了特征值的取值  i 代表特征值的取值
                    # 构建一个字典
                    #  p_x_y.index 代表着
                    self.x_types_proba[(idx, i, j)] = (p_x_y[i] + self.lambda_) / (
                                self.y_types_count[j] + p_x_y.shape[0] * self.lambda_)

    def predict(self, X_new):
        res = []
        for y in self.y_types:  # 遍历y的所有取值可能  为了求每个y取值的相关后验概率
            p_y = self.y_types_proba[y]  # 得到P（y=Ck）   self.y_types_proba y的各个取值的（类型） 概率 计算
            p_xy = 1
            for idx, x in enumerate(X_new):  # enumerate 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                #idx 代表第几个特征值 x代表特征值的取值
                p_xy *= self.x_types_proba[(idx, x, y)]  # 计算条件概率
            res.append(p_y * p_xy)
        for i in range(len(self.y_types)):
            print("{}对应的概率：{:.2%}".format(self.y_types[i], res[i]))
        # 返回最大后验概率对应的y值
        return self.y_types[np.argmax(res)]


def main():
    X_train = np.array([
        [1, "S"],
        [1, "M"],
        [1, "M"],
        [1, "S"],
        [1, "S"],
        [2, "S"],
        [2, "M"],
        [2, "M"],
        [2, "L"],
        [2, "L"],
        [3, "L"],
        [3, "M"],
        [3, "M"],
        [3, "L"],
        [3, "L"]
    ])
    y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    clf = NaiveBayes(lambda_=0.2)
    clf.fit(X_train, y_train)
    X_new = np.array([2, "S"])
    y_predict = clf.predict(X_new)
    print("{}被分类为：{}".format(X_new, y_predict))


if __name__ == '__main__':
    main()
