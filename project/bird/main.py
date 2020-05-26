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


bird = pd.read_csv("bird.csv").dropna().drop("id", axis=1)

bird["type"] = bird.type.apply(
    lambda t: t in ["SW", "W", "R"]
).astype(np.int)

data = bird.values
np.random.shuffle(data)

train_x = np.mat(data[:300, :-1])
train_y = np.mat(data[:300, -1]).T

test_x = np.mat(data[300:, :-1])
test_y = np.mat(data[300:, -1]).T

lr = LogisticRegression(Gradient())

lr.train(train_x, train_y)

p = lr.predict(test_x)
pred = (p > 0.5).astype(np.int)

print("正确率:{:.2f}%, 查准率:{:.2f}%, 查全率:{:.2f}%, ROC曲线积分:{:.3f}".format(
    accuracy_score(test_y, pred) * 100,
    precision_score(test_y, pred) * 100,
    recall_score(test_y, pred) * 100,
    roc_auc_score(test_y, pred) * 100
))
