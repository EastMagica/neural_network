#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/5/26 23:28
# @file    : regression.py
# @project : NNs
# software : PyCharm

import numpy as np


class LogisticRegression(object):
    def __init__(self, optimizer, iterations=100000):
        assert optimizer is not None
        self.optimizer = optimizer
        self.iterations = iterations
        self.weights = None

    def train(self, x, y):
        x = np.mat(np.c_[[1.0] * x.shape[0], x])
        self.weights = np.mat(np.random.normal(0, 0.01, size=x.shape[1])).T
        for i in range(self.iterations):
            p = self.predict(x, False)
            gradient = -x.T * (y - p) / x.shape[0]
            self.weights += self.optimizer.delta(gradient)
            if i % 100 == 0:
                cross_entropy = (-y.T * np.log(p) - (1.0 - y).T * np.log(1 - p)) / y.shape[0]
                accuracy = np.sum((p > 0.5).astype(np.int) == y).astype(np.int) / y.shape[0]
                print("iter:{:d}, 交叉熵:{:.6f}, 正确率:{:.2f}%".format(
                    i+1, cross_entropy[0, 0], accuracy * 100
                ))

    def predict(self, x, augment=True):
        if augment:
            x = np.mat(np.c_[[1.0] * x.shape[0], x])

        a = -np.matmul(x, self.weights)
        a[a > 1e2] = 1e2

        p = 1.0 / (1.0 + np.power(np.e, a))
        p[p >= 1.0] = 1.0 - 1e-10
        p[p <= 0.0] = 1e-10

        return p


