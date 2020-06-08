#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nbensemble import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.neural_network import *
from sklearn.model_selection import *


import pandas as pd
data = pd.read_csv('dataset.csv', index_col=0)
X, Y = data.iloc[:, :-1], data.iloc[:, -1].values
for i, y in enumerate(Y):
    if y>600:
        Y[i]=2
    elif y>500:
        Y[i]=1
    else:
        Y[i]=0


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

import numpy as np


keys = data.columns

key1=['用户实名制是否通过核实', '是否大学生客户', '是否黑名单客户', '是否4G不健康客户', '缴费用户当前是否欠费缴费',
'是否经常逛商场的人', '当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店', '当月是否看电影', '当月是否景点游览', '当月是否体育场馆消费']
key2 = ['用户年龄', '用户话费敏感度', '用户当月账户余额（元）', '近三个月月均商场出现次数',
'当月物流快递类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数', '当月旅游资讯类应用使用次数',
'用户网龄（月）', '用户最近一次缴费距今时长（月）',  '当月通话交往圈人数']
key3 = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）',
'当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数']
                                             


import time

estimators = [('bernoulli', BernoulliNB()), ('multinomial', MultinomialNB()), ('gauss', GaussianNB())]
nba1 = NBAdditive(estimators=estimators)

estimators = [('bernoulli', BernoulliNB()), ('tree', DecisionTreeClassifier()), ('gauss', GaussianNB())]
nba2 = NBAdditive(estimators=estimators)

models = [('NB组合（NB）', nba1), ('NB组合（非NB）', nba2),
('高斯NB', GaussianNB()), ('多项式NB', MultinomialNB()), ('决策树', DecisionTreeClassifier()), ('神经网络',  MLPClassifier(hidden_layer_sizes=(8,), max_iter=1800))]

perf = []
for name, model in models:
    time1 = time.perf_counter()
    if name.startswith('NB'):
        model.fit(X_train, Y_train, inds=[key1, key2, key3])
    else: 
        model.fit(X_train, Y_train)
    time2 = time.perf_counter()
    perf.append([name, model.score(X_test, Y_test), time2 - time1])


p = pd.DataFrame(data=perf, columns=('name', 'score', 'time'))
print(p)

# model = MLPClassifier()
# model.fit(X_train[key2], Y_train)
# print(model.score(X_test[key2], Y_test))

