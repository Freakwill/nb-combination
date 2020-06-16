#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The Naive Bayes' Ensemble Method

NBE Formula:
    ln p(c|x) sim sum_i ln f_{i,c}(xi) + (1-m) ln p(c),
where f_{i,c}(xi) is the prob of c when input is xi by model i,
m is the number of base estimators.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import _BaseNB
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble

class NBAdditive(_BaseHeterogeneousEnsemble, _BaseNB):
    def __init__(self, estimators, priors=None):
        """
        Arguments:
            estimators {list} -- list of classfier estimators
        """
        super(NBAdditive, self).__init__(estimators)
        names, self.estimators_ = zip(*estimators)
        self.priors = priors
        self.n_estimators = len(estimators)
        self.classes_ = None

    def fit(self, X, Y, inds):
        """
        fit the samples by nbe formula
        
        Arguments:
            X {array} -- input data
            Y {array} -- ouput data
            inds {list} -- list of indexes
        """

        import collections, threading
        if self.classes_ is None:
            self.class_count_ = collections.Counter(np.asarray(Y))
            self.n_classes_ = len(self.class_count_)
            self.classes_ = np.array(list(self.class_count_.keys()))
        classes = self.classes_.tolist()
        Y = list(map(classes.index, Y))

        if isinstance(X, pd.DataFrame):
            self.features_ = X.columns
            threads = [threading.Thread(target=model.fit, args=(X[ind], Y))
            for model, ind in zip(self.estimators_, inds)]
        elif isinstance(X, np.ndarray):
            threads = [threading.Thread(target=model.fit, args=(X[:, ind], Y))
            for model, ind in zip(self.estimators_, inds)]
        else:
            return self.fit(np.asarray(X), Y, inds)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if self.priors is None:
            epsilon = 0.1
            N = len(Y)
            logN = np.log(N + self.n_classes_ * epsilon)
            p = np.array([np.log(v + epsilon) - logN for k, v in enumerate(self.class_count_.values())])
            self.class_log_prior_ = p
            self.class_prior_ = np.exp(p)
            self.inds = inds

        return self


    def _joint_log_likelihood(self, X, inds=None):
        # by nbe formula, excluding a constant coef
        if inds is None:
            inds = self.inds
        p = 0
        if isinstance(X, pd.DataFrame):
            for model, ind in zip(self.estimators_, inds):
                p += model.predict_log_proba(X[ind])
        elif isinstance(X, np.ndarray):
            for model, ind in zip(self.estimators_, inds):
                p += model.predict_log_proba(X[:, ind])
        else:
            return self._joint_log_likelihood(np.asarray(X), inds)
         
        return p + (1-self.n_estimators) * self.class_log_prior_


    # def predict(self, X):
    #     return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


if __name__ == '__main__':
    X=np.array([[1,2,3.1],[2,2,2.3],[1,1,1.1], [1,1,2.1]])
    Y= [0,1,0,1]
    from sklearn.naive_bayes import *
    estimators = [('multinomial', MultinomialNB()), ('gauss', GaussianNB())]
    model = NBAdditive(estimators)
    model.fit(X, Y, [[0,1], [2]])
    print(model.predict(X))


