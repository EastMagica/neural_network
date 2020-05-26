#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/5/26 22:48
# @file    : gradient.py
# @project : NNs
# software : PyCharm

import numpy as np


# Gradient Descent Method
# =======================

class Gradient(object):
    r"""
    梯度下降法(Gradient Descent Method).
    """
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def delta(self, gradient):
        return -self.learning_rate * gradient


class DecayGradient(Gradient):
    r"""
    学习率调度 (Learning rate scheduling).

    学习率指数衰减策略 (Exponential decay scheduling).

    .. math::
        \eta^t = \eta^{\text{init}} \cdot 10^{-\frac{t}{r}}

    """
    def __init__(self, learning_rate=0.001, r=500):
        super().__init__(learning_rate)
        self.r = r
        self.global_steps = 0

    def delta(self, gradient):
        eta = self.learning_rate * 10 ** (-self.global_steps / self.r)
        self.global_steps += 1
        return -eta * gradient


class MomentumGradient(Gradient):
    r"""
    冲量法 (Momentum Method).

    引入摩擦减少震荡作用.
    """
    def __init__(self, learning_rate=0.001, beta=0.9):
        super().__init__(learning_rate)
        self.v = None
        self.beta = beta

    def delta(self, gradient):
        if self.v is None:
            self.v = np.mat(np.zeros(gradient.shape[0])).T
        self.v = self.beta * self.v - self.learning_rate * gradient
        return self.v


class AdaGradient(Gradient):
    """
    AdaGrad

    对函数局部二阶信息进行粗糙估计.
    """
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)
        self.s = None

    def delta(self, gradient):
        if self.s is None:
            self.s = np.mat(np.zeros(gradient.shape[0])).T
        self.s = self.s + np.power(gradient, 2)
        return -self.learning_rate * gradient / np.sqrt(self.s + 1e-10)


class RMSPropGradient(Gradient):
    """
    RMSProp

    相对于 AdaGrad, 使用局部梯度分量估计局部二阶信息.

    """
    def __init__(self, learning_rate=0.001, beta=0.9):
        super().__init__(learning_rate)
        self.s = None
        self.beta = beta

    def delta(self, gradient):
        if self.s is None:
            self.s = np.mat(np.zeros(gradient.shape[0])).T
        self.s = self.beta * self.s + (1 - self.beta) * np.power(gradient, 2)
        return -self.learning_rate * gradient / np.sqrt(self.s + 1e-10)


class AdamGradient(Gradient):
    """
    (Adaptive Moment Estimation).

    结合 冲量法 和 RMSProp 的思想.

    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.99):
        super().__init__(learning_rate)
        self.s = None
        self.v = None
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.global_steps = 0

    def delta(self, gradient):
        if self.s is None or self.v is None:
            self.s = np.mat(np.zeros(gradient.shape[0])).T
            self.v = np.mat(np.zeros(gradient.shape[0])).T
        self.v = self.beta_1 * self.v + (1 - self.beta_1) * gradient
        self.s = self.beta_2 * self.s + (1 - self.beta_2) * np.power(gradient, 2)
        self.v = self.v / (1 - self.beta_1 ** (self.global_steps + 1))
        self.s = self.s / (1 - self.beta_2 ** (self.global_steps + 1))
        self.global_steps += 1
        return -self.learning_rate * self.v / np.sqrt(self.s + 1e-10)


