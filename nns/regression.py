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

    def predict(self, x, augment=True):
        """
        预测函数.

        Parameters
        ----------
        x: ndarray
            矩阵形状 (n_samples, n_features).
        augment: bool
            是否在特征矩阵前添加一列常量 1.

        Returns
        -------
        p: ndarray
            预测正类概率, 矩阵形状 (n_samples, 1).
        """
        if augment:
            # 在 x 前添加一列常数 1.0, 作为偏置值的输入
            x = np.mat(np.c_[[1.0] * x.shape[0], x])

        a = -np.matmul(x, self.weights)
        a[a > 1e2] = 1e2  # 防止数值过大

        p = 1.0 / (1.0 + np.power(np.e, a))
        # 裁剪概率值, 保证其为合法概率
        p[p >= 1.0] = 1.0 - 1e-10
        p[p <= 0.0] = 1e-10

        return p

    def train(self, x, y):
        """
        训练函数.

        Parameters
        ----------
        x: ndarray
            矩阵形状 (n_samples, n_features).
        y: ndarray
            样本标签, 矩阵形状 (n_samples, 1), 正类为 1, 负类为 0.
        """
        # 样本矩阵前添加一列常数, 作为偏置值输入, 简化公式
        x = np.mat(np.c_[[1.0] * x.shape[0], x])

        # 根据样本矩阵的列数随机初始化权值
        # 偏置值纳入权值向量, 相当于第一个权值
        # 权值向量为 n_features+1 维向量
        # 每个分量以 0 为均值, 0.01 标准差的正态分布初始化
        self.weights = np.mat(np.random.normal(0, 0.01, size=x.shape[1])).T

        for i in range(self.iterations):
            # 计算当前模型对训练样本的输出
            p = self.predict(x, False)

            # 交叉熵损失对参数模型的梯度
            gradient = -x.T * (y - p) / x.shape[0]

            # 更新模型参数
            self.weights += self.optimizer.delta(gradient)

            # 评估当前模型, 打印训练信息
            if i % 100 == 0:
                # 交叉熵损失
                cross_entropy = (-y.T * np.log(p) - (1.0 - y).T * np.log(1 - p)) / y.shape[0]

                # 正确率
                accuracy = np.sum((p > 0.5).astype(np.int) == y).astype(np.int) / y.shape[0]

                print("iter:{:d}, 交叉熵:{:.6f}, 正确率:{:.2f}%".format(
                    i + 1, cross_entropy[0, 0], accuracy * 100
                ))
