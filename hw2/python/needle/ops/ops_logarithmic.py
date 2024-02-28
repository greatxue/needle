from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, self.axes, keepdims=True)
        max_Z_reduced = array_api.max(Z, self.axes)
        # The maximum is deducted in advance in case of numerical overflow.
        return array_api.log(array_api.sum(array_api.exp(Z - max_Z), self.axes)) + max_Z_reduced
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_Z = Z.realize_cached_data().max(self.axes, keepdims=True)
        exp_Z = exp(Z - max_Z)
        sum_exp_Z = summation(exp_Z, self.axes)
        grad_sum_exp_Z = out_grad / sum_exp_Z
        expand_shape = list(Z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        # The index below will not cause out-of-range error, as expand_shape is of the same size as 
        #  dimensions of Z, or even less if axes are specified.
        for axis in axes:
            expand_shape[axis] = 1
        # grad_exp_Z needs to reshape to 1-processed shape and then broadcast to the shape of Z.
        grad_exp_Z = grad_sum_exp_Z.reshape(expand_shape).broadcast_to(Z.shape)

        return grad_exp_Z * exp_Z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

