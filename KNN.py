# 使用线性扫描算法实现
# 使用欧式距离 取前k个最小距离的类型  取频率最高的类型

import numpy as np
from collections import Counter


class KNN:
    def __init__(self, X, Y, k=3):
        self.X = X
        self.Y = Y
        self.k = k

    def predict(self, X_new):
        # 计算欧式距离 dist_list 中既有距离，也有对应的Y
        dist_list = [(np.linalg.norm(X_new - self.X[i], ord=2), self.Y[i])
                     for i in range(self.X.shape[0])]
        # 对所有距离排序
        dist_list.sort(key=lambda x: x[0])
        print(dist_list)
        # 取前k个最小距离
        y_list = [dist_list[i][-1] for i in range(self.k)]
        y_count = Counter(y_list).most_common()
        return y_count[0][0]


if __name__ == '__main__':
    # 训练数据
    X = np.array([[5, 4],
                  [9, 6],
                  [4, 7],
                  [2, 3],
                  [8, 1],
                  [7, 2]])
    Y = np.array([1, 1, 1, -1, -1, -1])
    # 测试数据
    X_new = np.array([[5, 3]])

    for k in range(1, 6, 2):  # 验证不同的k值对结果的影响
    # 构建KNN实例
        clf = KNN(X, Y, k=k)
        y_predict = clf.predict(X_new)
        print(k,"    ",y_predict)
