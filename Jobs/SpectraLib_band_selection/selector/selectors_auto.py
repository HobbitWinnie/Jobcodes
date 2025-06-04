# selectors_auto.py

import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

def rfe_band_select(X, y, n_features=3, estimator=None):
    """递归特征剔除（RFE）选波段"""
    if estimator is None:
        estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features).fit(X, y)
    chosen = np.where(selector.support_)[0]
    return chosen, selector.ranking_

def rf_band_select(X, y, n_features=3):
    """随机森林重要性选波段"""
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    imp = rf.feature_importances_
    chosen = np.argsort(-imp)[:n_features]
    return chosen, imp

def lsvc_band_select(X, y, n_features=3):
    """稀疏线性SVM重要性选波段"""
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=3000)
    lsvc.fit(X, y)
    abs_coefs = np.abs(lsvc.coef_).reshape(-1)
    chosen = np.argsort(-abs_coefs)[:n_features]
    return chosen, abs_coefs