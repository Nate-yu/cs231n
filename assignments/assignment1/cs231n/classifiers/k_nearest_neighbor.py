from builtins import range
from builtins import object
import numpy as np
# from past.builtins import xrange # past已经在Python3中了，不需要再导入


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练分类器。对于k近邻，这只是记忆训练数据。

        输入:
        - X： 包含训练数据，形状为(num_train,D)的numpy数组，该训练数据由每个维度为D的num_train样本组成。
        - y： 一个包含训练标签，形状为(N,)的numpy数组，其中y[i]是X[i]的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        使用此分类器预测测试数据的标签。

        输入:
        - X：形状为(num_test,D)的numpy数组，包含由每个维度为D的num_test样本组成的测试数据。
        - k：投票支持预测标签的最近邻居的数量。
        - num_loops：确定使用哪个实现来计算训练点和测试点之间的距离。

        返回:
        - y： 形状（num_test）的numpy数组，包含测试数据的预测标签，其中y[i]是测试点X[i]的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        计算X中的每个测试点与self中的每个训练点之间的距离。X_train在训练数据和测试数据上使用嵌套循环。

        输入:
        - X： 包含测试数据的形状为(num_test,D)的numpy数组。

        返回:
        - dists：形状为(num_test,num_train)的numpy数组，其中dists[i,j]是第i个测试点和第j个训练点之间的欧几里得距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # 计算第i个测试点和第j个测试点之间的l2距离             #
                # 训练点，并将结果存储在dists[i，j]中。              #
                # 你不应该在维度上使用循环，也不要使用np.linalg.norm（）。     #
                #####################################################################
                # *****代码的开头（不要删除/修改此行）*****
                # 第i个X_test的图像和第J个X_train的图像的距离存放在dists[i,j]中
                dists[i,j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

                # *****代码末尾（不要删除/修改此行）*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        计算X中的每个测试点与self中的每个训练点之间的距离。X_train在测试数据上使用单个循环。

        输入/输出：与compute_dinstances _two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # 计算第i个测试点与所有训练点之间的l2距离，并将结果存储在dists[i，：]中。                       #
            # 不要使用np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 对每一行作L2距离计算
            dists[i,:] = np.sqrt(np.sum(np.square(X[i]-self.X_train), axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        计算X中的每个测试点与self中的每个训练点之间的距离。X_train不使用显式循环。

        输入/输出：与compute_dinstances _two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                #                                 #
        # 在不使用任何显式循环的情况下计算所有测试点和所有训练点之间的l2距离，#
        # 并将结果存储在dists中。                       #                                         #
        #                                                               #
        # 你应该仅使用基本的数组操作来实现此函数；                      #
        # 特别是，您不应该使用scipy中的函数，                 #
        # 不要使用 np.linalg.norm().                     #                        #
        #                                    #                                   #
        # 提示: 尝试使用矩阵乘法和两个广播和来公式化l2距离。         #                                    #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dist_a = np.sum(X**2, axis=1, keepdims=True) # (M,1)
        dist_b = np.sum(self.X_train**2, axis=1) # (N,)
        dist_c = -2 * X.dot(self.X_train.T) # (M,d) * (d,N) = (M,N)
        dists = (dist_a + dist_b + dist_c) ** 0.5

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        给定测试点和训练点之间的距离矩阵，预测每个测试点的标签。

        输入：
        - dists：形状为(num_test,num_train)的numpy数组，其中dists[i,j]给出了第i个测试点和第j个训练点之间的距离。

        返回:
        - y： 形状为(num_test)的numpy数组，包含测试数据的预测标签，其中y[i]是测试点X[i]的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 长度为k的列表，存储到第i个测试点的k个最近邻居的标签。
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # 使用距离矩阵找到第i个测试点的k个最近邻居，              #
            # 并使用self.y_train来查找这些邻居的标签。               #
            # 将这些标签保存在closest_y中。                     #
            # 提示: 查找函数numpy.argsort.                     # 
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # numpy.argsort()函数用于使用关键字kind指定的算法沿给定轴执行间接排序。
            # 它对数组进行升序排序。返回排序后元素在原数组中的索引。
            dist_idx = np.argsort(dists[i])
            closest_y = self.y_train[dist_idx]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # 既然你已经找到了k个最近邻居的标签,    #
            # 你需要在最接近y个标签的列表中找到最常见的标签。   #
            # 将此标签存储在y_pred[i]中。选择较小的标签，打破束缚。     #                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # y_pred[i] = np.argmax(np.bincount(closest_y))
            bincount = np.zeros(10)
            bin_sum = np.zeros(10) + np.inf
            for j in closest_y[:k]:
              bincount[j] += 1

            max_num = np.max(bincount)
            class_idx = np.where(bincount == max_num)[0] # 取出k数最大的类别，可能有多个
            if len(class_idx) > 1: # 当有多个类别的图片贡献的张数都相同时，我们将平均距离最近的类别作为预测类别可以提高模型准确率
              # 相比将第一个类别作为预测类别而言(代码短)，这种优化方法实测准确率可以提升一个百分点。
              dist_idx_k = dist_idx[:k] # 前k个距离的下标
              dist_k = dists[i, dist_idx_k] # 前k个距离
              for j in class_idx: # 将每个最大类别的欧几里得距离求总和
                idx_j = np.where(closest_y[:k] == j)[0] # 前k个类别的标签
                bin_sum[j] = np.sum(dist_k[idx_j])
              y_pred[i] = np.argmin(bin_sum) # 取出距离最近的那个类作为预测值
            else:
              y_pred[i] = class_idx[0]


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
