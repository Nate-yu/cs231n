from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from ..classifiers.linear_svm import *
from ..classifiers.softmax import *
# from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        使用随机梯度下降训练该线性分类器。

        Inputs:
        - X：一个包含训练数据的形状(N,D)的numpy数组；存在N个训练样本，每个训练样本具有维度D。
        - y：一个包含训练标签的形状(N,)的numpy数组；y[i]=c意味着X[i]对于c类具有标签0 <= c <C。
        - learning_rate：优化的（浮动）学习速率。
        - reg:（float）正则化强度。
        - num_iters：（整数）优化时要执行的步骤数
        - batch_size: （整数）在每个步骤中使用的训练示例的数量。
        - verbose: （布尔值）如果为true，则在优化期间打印进度。

        Outputs:
        包含每次训练迭代中损失函数值的列表。
        """
        num_train, dim = X.shape # 获取训练样本数量 num_train 和特征维度 dim
        num_classes = (np.max(y) + 1)  # 获取类别数量，假设y取值0…K-1，其中K是类的数量
        if self.W is None: # 如果权重矩阵 self.W 未初始化，使用一个小的随机值进行初始化
            # 延迟初始化W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 运行随机梯度下降以优化W
        loss_history = []
        for it in range(num_iters):
            # 对于训练迭代的次数，执行以下操作
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                  #                               
            # 对训练数据中的batch_size元素及其对应标签进行采样，以在本轮梯度下降中使用。#
            # 将数据存储在X_batch中，并将其相应的标签存储在y_batch中；         #
            # 采样后X_batch应具有shape（batch_size，dim），y_batch应为shape（batch_size，）#
            #                                       #
            # 提示：使用np.random.choice生成索引。使用替换进行采样比不使用替换进行采样更快。#
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # 随机选择 batch_size 个样本的索引，用于当前训练迭代
            choice_idxs = np.random.choice(num_train,batch_size)
            # 根据上述索引获取对应的训练数据和标签。
            X_batch = X[choice_idxs]
            y_batch = y[choice_idxs]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # 计算当前训练迭代的损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 执行参数更新
            #########################################################################
            # TODO:                                #
            # 使用梯度和学习率更新权重。                      #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.W -= learning_rate * grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # 如果设置了 verbose 并且当前迭代是100的倍数，打印当前迭代的损失值。
            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        使用此线性分类器的训练权重来预测数据点的标签。

        Inputs:
        - X: 一个包含训练数据的形状（N，D）的numpy数组；存在N个训练样本，每个训练样本具有维度D。

        Returns:
        - y_pred: X.y_pred中数据的预测标签是长度为N的一维数组，每个元素都是给出预测类的整数。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # 实现此方法。将预测的标签存储在y_pred中。           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算每个样本的得分，scores 是一个矩阵，其中每行对应一个样本，每列对应一个类别。
        scores = X.dot(self.W) 
        # 对于每个样本，选取具有最高得分的类别作为预测类别。axis=1 表示在每行中找到最大值的索引，将这些索引存储在 y_pred 中。
        y_pred = np.argmax(scores,axis = 1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        计算损失函数及其导数。
        子类将覆盖此项。

        Inputs:
        - X_batch: 一个形状为(N, D)的numpy数组，包含N个数据点的小批量；每个点具有尺寸D。
        - y_batch: 包含迷你批次标签的形状（N，）的numpy数组。
        - reg: （浮动）正则化强度。

        Returns: 一个元组，包含：
        - 单个浮点数的损失
        - 相对于self.W。与W形状相同的阵列
        """
        pass


class LinearSVM(LinearClassifier):
    """ 使用多类SVM损失函数的子类 """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
