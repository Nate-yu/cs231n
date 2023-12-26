from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    结构化SVM损失函数，天真实现（带循环）。

    输入有维度D，有C类，我们对N个例子的迷你批次进行运算。

    Inputs:
    - W： 包含权重的形状（D，C）的numpy数组。
    - X： 一个数字数组，形状为（N，D），包含一个小数据块。
    - y： 一个包含训练标签的形状（N，）的numpy数组；y[i]=c意味着X[i]具有标签c，其中0 <= c < C。
    - reg: （浮动）正则化强度

    返回以下元组：
    - 单浮点型loss
    - 相对于权重W的梯度；与W形状相同的阵列
    """
    dW = np.zeros(W.shape)  # 初始化梯度矩阵为零，其形状与权重矩阵 W 相同

    # 计算损失和梯度
    num_classes = W.shape[1] # 类别数量
    num_train = X.shape[0] # 训练集中样本的数量
    loss = 0.0
    for i in range(num_train): # 遍历训练集中的每个样本
        scores = X[i].dot(W) # 计算每个类别的分数
        correct_class_score = scores[y[i]] # 获取正确类别的分数
        for j in range(num_classes): # 遍历每个类别
            if j == y[i]: # 遍历到标签类时不做损失和梯度的累加，对于正确类别的分数不参与损失的计算
                continue
            margin = scores[j] - correct_class_score + 1  # 计算间隔，注意 delta = 1
            if margin > 0: # 如果间隔大于零，说明存在损失
                loss += margin # 累加损失
                dW[:, y[i]] += -X[i].T # 更新梯度，对正确类别的列进行累加
                dW[:, j] += X[i].T # 更新梯度，对错误类别的列进行累加
    

    # 现在，损失是所有训练示例的总和，但我们希望它是一个平均值，所以我们除以num_train。
    loss /= num_train

    # 将正则化添加到损失中
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # 计算损失函数的梯度并将其存储为dW。                    #
    # 相反，首先计算损失，然后计算导数，在计算损失的同时计算导数可能更简单。  #
    # 因此，您可能需要修改上面的一些代码来计算梯度。              #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train # 对梯度进行平均
    dW += 2 * reg * W # 添加正则化项到梯度中

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    结构化SVM损失函数，矢量化实现。
    输入和输出与svm_loss_naive相同。
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # 将梯度初始化为与W形状相同的全0矩阵

    #############################################################################
    # TODO:                                                                     #
    # 实现矢量化版本的结构化SVM损失，将结果存储在损失中。                           # 
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = len(y)
    scores = X.dot(W) # 计算分数矩阵 scores
    scores -= scores[range(N), [y]].T # 通过广播操作减去正确类别的分数
    scores += 1 # 加上边界值 1
    scores[range(N), y] = 0 # 将正确类别的分数设置为 0
    # 将scores矩阵中的每个元素与0比较，取较大的值。这一操作的结果是生成一个新的矩阵margin，其中所有小于等于0的元素都变成了0，而大于0的元素保持不变。
    # 这样做的目的是计算每个样本在每个类别上的间隔（即正类别的得分减去其他类别的得分），并将小于等于0的间隔变为0，符合支持向量机损失函数的形式。
    margin = np.maximum(0, scores) 
    loss = np.sum(margin) / N + reg * np.sum(W**2) # 计算损失，将所有间隔相加并除以样本数量，再加上正则化项

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # 为结构化SVM损失实现梯度的矢量化版本，将结果存储在dW中。                        #                                  
    #                                                                           #
    # 提示：与其从头开始计算梯度，不如重用一些用于计算损失的中间值。                  #            
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 创建一个与 margin 大小相同的矩阵 ds，用于记录有损失的地方为 1，其他地方为 0
    ds = np.zeros_like(margin) 
    ds[margin > 0] = 1
    # 然后，对于每个样本，将正确类别对应的列减去该行上的 ds 的总和
    ds[range(N), y] -= np.sum(ds, axis=1)
    # 最后，计算梯度矩阵 dW，加上正则化项
    dW += X.T.dot(ds)
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
