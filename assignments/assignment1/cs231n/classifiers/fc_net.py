from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    采用模块化层设计的两层全连接神经网络，具有非线性的ReLU和softmax loss。我们假设输入维度为D，隐藏维度为H，并对C个类进行分类。
    架构应该是 affine - relu - affine - softmax.

    注意，这个类不实现梯度下降;相反，它将与负责运行优化的单独求解器对象交互。

    模型的可学习参数存储在字典本身中。将参数名映射到numpy数组的参数。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
      
        """
        初始化新网络。

        Inputs:
        - input_dim: 一个给出输入大小的整数
        - hidden_dim: 一个整数，表示隐藏层的大小
        - num_classes: 一个整数，表示要分类的类的数量
        - weight_scale: 给出权重随机初始化的标准差的标量。
        - reg: 给出L2正则化强度的标量。
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: 初始化两层网的权重和偏差。                                           #
        # 权重应从以0.0为中心的高斯值初始化，标准偏差等于weight_scale，偏差应初始化为零。
        # 所有权重和偏差应存储在字典self-params中，第一层权重和偏差使用关键字“W1”和“b1”，第二层权重和偏差使用密钥“W2”和“b2”。#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(loc=0.0,scale=weight_scale,size=(input_dim,hidden_dim))
        self.params['W2'] = np.random.normal(loc=0.0,scale=weight_scale,size=(hidden_dim,num_classes))
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['b2'] = np.zeros((num_classes,))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #            END OF YOUR CODE                                              #
        ############################################################################

    def loss(self, X, y=None):
        """
        计算小批量数据的损失和梯度。

        Inputs:
        - X: 形状为（N，d_1，…，d_k）的输入数据数组
        - y: 标签数组，形状为（N，）。y[i]给出了X[i]的标签

        Returns:
        如果y为None，则运行模型的测试时间前向传递并返回：
        - scores：给出分类分数的形状（N，C）的数组，其中scores[i，C]是X[i]和类C的分类分数。

        如果y不为None，则运行训练时间前后传递，并返回以下元组：
        - loss: 给出损失的标量值
        - grads: 字典，具有与self.params相同的键，将参数名称映射到相对于这些参数的损失梯度。
        """
        scores = None
        ############################################################################
        # 实现两层网络的正向传递，计算X的类分数并将其存储在分数变量中。       #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']


        affine_1_out,affine_1_cache = affine_forward(X,W1,b1)
        relu_out,relu_cache = relu_forward(affine_1_out)
        affine_2_out,affine_2_cache = affine_forward(relu_out,W2,b2)
        # 不过softmax，因为softmax计算loss，这里只需要scores
        scores = affine_2_out

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 如果y为None，则我们处于测试模式，因此只返回分数
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: 实现双层网络的反向传播。                     #
        # 将损失存储在损失变量中，将梯度存储在梯度字典中。            #
        # 使用softmax计算数据损失，并确保grades[k]保持self.params[k]的梯度。
        # 不要忘记添加L2正则化！                   #
        #                                      #
        # NOTE: 为了确保您的实现与我们的匹配并通过自动测试，请确保L2正则化包含0.5的因子，以简化梯度的表达式。#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 使用softmax计算loss和grad
        loss,d_affine_2_out = softmax_loss(affine_2_out,y)
        # 加上正则项
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # 计算梯度
        d_relu_out,dW_2,dB_2 = affine_backward(d_affine_2_out,affine_2_cache)
        d_affine_1_out = relu_backward(d_relu_out,relu_cache)
        dX,dW_1,dB_1 = affine_backward(d_affine_1_out,affine_1_cache)

        dW_1 += self.reg * W1
        dW_2 += self.reg * W2

        # 保存梯度
        grads['W1'] = dW_1
        grads['W2'] = dW_2
        grads['b1'] = dB_1
        grads['b2'] = dB_2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #   END OF YOUR CODE                         #
        ############################################################################

        return loss, grads
