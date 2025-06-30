import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


def rf_band_select(X, y, n_features=3):
    """基于RF重要性，使用每个特征无NaN子集得分"""
    n_bands = X.shape[1]
    importances = np.zeros(n_bands)
    for i in range(n_bands):
        mask = ~np.isnan(X[:, i])
        X_sub = X[mask, i].reshape(-1, 1)
        y_sub = y[mask]
        if X_sub.shape[0] < 2 or len(np.unique(y_sub)) < 2:
            importances[i] = 0  # 样本太少或只有一个类别
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(X_sub, y_sub)
            importances[i] = clf.feature_importances_[0]
    chosen = np.argsort(-importances)[:n_features]
    return chosen, importances

def lsvc_band_select(X, y, n_features=3):
    """基于L1 SVM重要性，按每个特征无NaN子样本得分"""
    n_bands = X.shape[1]
    coefs = np.zeros(n_bands)
    for i in range(n_bands):
        mask = ~np.isnan(X[:, i])
        X_sub = X[mask, i].reshape(-1, 1)
        y_sub = y[mask]
        if X_sub.shape[0] < 2 or len(np.unique(y_sub)) < 2:
            coefs[i] = 0
        else:
            clf = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000, random_state=0)
            try:
                clf.fit(X_sub, y_sub)
                coefs[i] = np.abs(clf.coef_[0, 0])
            except Exception:
                coefs[i] = 0
    chosen = np.argsort(-coefs)[:n_features]
    return chosen, coefs

def rfe_band_select(X, y, n_features=3, estimator=None):
    """
    RFE只能用完整无NaN样本，因此这里自动使用全部波段都无NaN的行
    """
    mask = ~np.isnan(X).any(axis=1)
    X_clean, y_clean = X[mask], y[mask]
    if estimator is None:
        estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features).fit(X_clean, y_clean)
    chosen = np.where(selector.support_)[0]
    return chosen, selector.ranking_