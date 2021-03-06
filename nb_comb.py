#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The Naive Bayes' Combination Algorithm

NBE Formula:
    ln p(c|x) sim sum_i ln f_{i,c}(xi) + (1-m) ln p(c),
where f_{i,c}(xi) is the prob of c when input is xi by model i,
m is the number of base estimators.

Created by William Song
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import _BaseNB
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.model_selection import train_test_split

def _log_proba(model, x):
    # log p(c|x)
    if hasattr(model, 'predict_log_proba'):
        return model.predict_log_proba(x)
    elif hasattr(model, 'predict_proba'):
        return np.log(model.predict_proba(x))
    else:
        raise Exception(f"The model {model} does not have attribute 'predict_log_proba' or 'predict_proba'!")


class NBAdditive(_BaseHeterogeneousEnsemble, _BaseNB):
    def __init__(self, estimators, priors=None):
        """
        Arguments:
            estimators {list} -- list of basic classfiers
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

    # @property
    # def class_prior_(self):
    #     return np.exp(self._class_log_prior_)
    

    def _joint_log_likelihood(self, X, inds=None):
        # by nbe formula, excluding a constant coef
        if inds is None:
            inds = self.inds
        p = 0
        if isinstance(X, pd.DataFrame):
            for model, ind in zip(self.estimators_, inds):
                p += _log_proba(model, X[ind])
        elif isinstance(X, np.ndarray):
            for model, ind in zip(self.estimators_, inds):
                p += _log_proba(model, X[:, ind])
        else:
            return self._joint_log_likelihood(np.asarray(X), inds)
         
        return p + (1-self.n_estimators) * self.class_log_prior_


    # def predict(self, X):
    #     return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

class SANBAdditive(NBAdditive):

    def fit(self, X, Y, inds, search=True):
        super(SANBAdditive, self).fit(X, Y, inds)
        if search:
            self.coefs = self.get_coefs(X, Y, inds)
        return self

    def get_coefs(self, X, Y, inds):
        from sko.SA import SA_TSP
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3)
        self.fit(X_train, Y_train, inds, search=False)
        self.init_coefs = np.ones((self.n_estimators, self.n_classes_))
        def goal(coefs):
            self.coefs = coefs.reshape((self.n_estimators, self.n_classes_))
            return self.score(X_valid, Y_valid)
        sa_tsp = SA_TSP(func=goal, x0=self.init_coefs.ravel(), T_max=100, T_min=1, L=20)
        best_points, _ = sa_tsp.run()
        return best_points.reshape((self.n_estimators, self.n_classes_))

    def _joint_log_likelihood(self, X, inds=None):
        # by nbe formula, excluding a constant coef
        if inds is None:
            inds = self.inds
        p = 0
        if isinstance(X, pd.DataFrame):
            for model, ind, coef in zip(self.estimators_, inds, self.coefs):
                p += coef * _log_proba(model, X[ind])
        elif isinstance(X, np.ndarray):
            for model, ind, coef in zip(self.estimators_, inds, self.coefs):
                p += coef * _log_proba(model, X[:, ind])
        else:
            return self._joint_log_likelihood(np.asarray(X), inds)
         
        return p + (1-self.n_estimators) * self.class_log_prior_



if __name__ == '__main__':
    X=np.array([[1,2,3.1],[2,2,2.3],[1,1,1.1], [1,1,2.1]])
    Y= [0,1,0,1]
    from sklearn.naive_bayes import *
    estimators = [('multinomial', MultinomialNB()), ('gauss', GaussianNB())]
    model = NBAdditive(estimators)
    model.fit(X, Y, [[0,1], [2]])
    print(model.predict(X))


