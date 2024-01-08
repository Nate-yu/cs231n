from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax损失函数，带循环的简单实现

    输入有维度D，有C个类，我们在N个例子的小批量上操作。

    Inputs:
    - W: 包含权重的形状（D，C）的numpy数组。
    - X: 一个数字数组，形状为（N，D），包含一个小数据块。
    - y: 一个包含训练标签的形状（N，）的numpy数组；y[i]=c意味着X[i]具有标签c，其中0&lt;=c&lt;c。
    - reg: （浮动）正则化强度

    Returns a tuple of:
    - 单浮损失
    - 相对于权重W的梯度；与W形状相同的阵列
    """
    # 将损耗和梯度初始化为零。 
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: 使用显式循环计算softmax损失及其梯度。                                  #
    # 将损失存储在损失中，将梯度存储在dW中。                                        #
    # 如果你在这里不小心，很容易遇到数值不稳定的情况。不要忘记正则化！                #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0] # 图片的总数：500
    C = W.shape[1] # 类别数量：10
    # print("N={},C={}".format(N,C))

    scores = X.dot(W)
    # 对scores做指数运算，其中出现的np.max是为了让数值更加稳定(防止指数爆炸)
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    # 归一化操作：将每个标签类的分数除以总和，取-log并求和再除以N最后加上正则化项
    loss = np.sum(-np.log(scores_exp[range(N), y] / np.sum(scores_exp, axis=1))) / N + reg * np.sum(W**2)
    for i in range(N):
      dW[:, y[i]] -= X[i]
      for j in range(C):
        dW[:, j] += X[i] * scores_exp[i,j] / np.sum(scores_exp[i])
    dW /= N
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax损失函数，矢量化版本。

    输入和输出与softmax_loss_naive相同。
    """
    # 将损耗和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: 不使用显式循环计算softmax损失及其梯度。                                #
    # 将损失存储在损失中，将梯度存储在dW中。                                        #
    # 如果你在这里不小心，很容易遇到数值不稳定的情况。不要忘记正则化！                #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    scores = X.dot(W)
    # 对scores做指数运算，其中出现的np.max是为了让数值更加稳定(防止指数爆炸)
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    # 归一化操作：将每个标签类的分数除以总和，取-log并求和再除以N最后加上正则化项
    loss = np.sum(-np.log(scores_exp[range(N), y] / np.sum(scores_exp, axis=1))) / N + reg * np.sum(W**2)
    ds = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    ds[range(N),y] -= 1
    dW = X.T.dot(ds)
    
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
