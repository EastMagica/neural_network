#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/5/26 23:43
# @file    : main.py
# @project : NNs
# software : PyCharm

import numpy as np
import pandas as pd

from nns.grad import *
from nns.regression import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 读取数据
bird = pd.read_csv("bird.csv").dropna().drop("id", axis=1)

# 根据标签是否属于 "SW", "W", "R" 三类
# 构造二分类 0/1 标签
bird["type"] = bird.type.apply(
    lambda t: t in ["SW", "W", "R"]
).astype(np.int)

# 随机洗牌
data = bird.values
np.random.shuffle(data)

# 前300个样本作为训练集
train_x = np.mat(data[:300, :-1])
train_y = np.mat(data[:300, -1]).T

# 其余样本作为测试集
test_x = np.mat(data[300:, :-1])
test_y = np.mat(data[300:, -1]).T

# 构造逻辑回归对象
# 选择Adam优化器, 超参数取默认值
lr = LogisticRegression(AdamGradient())

# 在训练集上训练
lr.train(train_x, train_y)

# 对测试集进行测试
# 模型预测的正类概率
p = lr.predict(test_x)
# 以0.5为阈值, 模型预测的类别
pred = (p > 0.5).astype(np.int)

print("正确率:{:.2f}%, 查准率:{:.2f}%, 查全率:{:.2f}%, ROC曲线面积:{:.3f}".format(
    accuracy_score(test_y, pred) * 100,
    precision_score(test_y, pred) * 100,
    recall_score(test_y, pred) * 100,
    roc_auc_score(test_y, pred) * 100
))
