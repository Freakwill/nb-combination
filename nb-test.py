#!/usr/bin/env python

"""
图片分类举例

图片分解成高频部分和低频部分。高频部分进行二值化处理作为离散变量。
"""

import pathlib

from PIL import Image
from scipy.signal import convolve2d
from nb_comb import *

# requirements: Pillow, nb_comb

# 输入图片文件路径
# 路径下，文件夹名设为类型名，其中的图片为该类图片。见文件夹images组织

IMAGE_PATH = pathlib.Path('images/')

sobel1 = np.array([[1,2,1],[0,0,0], [-1,-2,-1]])
sobel2 = sobel1.T
sobel3 = np.array([[0,0,0],[0,1,0], [0,0,0]]) - sobel1 - sobel2

# generate data
X1 = [] # input data, seperated to three parts
X2 = []
X3 = []
Y = []  # labels
classes = [] # the names of classes
k = 0        # the indexes of classes
for folder in IMAGE_PATH.iterdir():
    if folder.is_dir():
        for file in folder.iterdir():
            if file.suffix in {'.jpg', '.jpeg'}:
                im = Image.open(file)
                im  = np.asarray(im.resize((100,100)), dtype=np.float64).reshape((100,300))
                X1.append(np.ravel(convolve2d(im, sobel1)[::5,::5])>128)  # discrete part1
                X2.append(np.ravel(convolve2d(im, sobel2)[::5,::5])>128)  # discrete part2
                X3.append(np.ravel(convolve2d(im, sobel3)[::10,::10]))    # continous part3
                Y.append(k)
    classes.append(folder.stem)
    k += 1
X1 = np.asarray(X1)
X2 = np.asarray(X2)
X3 = np.asarray(X3)
Y = np.asarray(Y)

from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.model_selection import *


model = NBAdditive(estimators=[('Bernoulli NB', BernoulliNB()),
    ('Bernoulli NB', BernoulliNB()),
    ('Logistic Regression', LogisticRegression()), ])

X1train, X1test, X2train, X2test, X3train, X3test, Ytrain, Ytest = train_test_split(X1, X2, X3, Y, test_size=0.2)

model.ezfit([X1train, X2train, X3train], Ytrain)
print(f'test error: {model.ezscore([X1test,X2test, X3test], Ytest):.4}')
