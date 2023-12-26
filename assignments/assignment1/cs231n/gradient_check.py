from __future__ import print_function
from builtins import range
# from past.builtins import xrange

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a
    blob where outputs will be written. For example, f might be called like:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs:
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    对一些随机元素进行采样，只返回该维度的数值。用于检查数值梯度和解析梯度之间的相对误差
    f: 一个接受输入向量 x 并返回一个标量的函数
    x: 输入向量
    analytic_grad: 解析梯度
    num_checks: 进行梯度检查的次数
    h: 用于计算数值梯度的微小增量。
    """

    for i in range(num_checks):
        # 从输入向量的每个维度中随机选择一个索引，形成一个元组 ix。这样就在输入向量中随机选择了一个位置进行数值梯度检查。
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix] # 保存选择的维度的原始值。
        x[ix] = oldval + h  # 将选择的维度增加微小的增量 h
        fxph = f(x)  # 计算修改后的输入向量 x 对应的函数值
        x[ix] = oldval - h  # 将选择的维度减小微小的增量 h
        fxmh = f(x)  # 计算修改后的输入向量 x 对应的函数值
        x[ix] = oldval  # 还原选择的维度的原始值

        grad_numerical = (fxph - fxmh) / (2 * h) # 计算数值梯度，通过中心差分法得到(f(x + h) - f(x - h)) / (2h)
        grad_analytic = analytic_grad[ix] # 从解析梯度中获取相应的值
        # 计算相对误差，使用绝对值，分母加上一个很小的数值，避免分母为零
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        # 打印数值梯度、解析梯度以及相对误差。这样可以用于观察数值梯度和解析梯度之间的差异。
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
