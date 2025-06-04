import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def mutual_info_score(X, y):
    """
    多波段组合与类别的互信息
    X: (n_samples, n_selected_bands)
    y: (n_samples,)
    """
    # MI对每个特征，组合时直接平均
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    return np.mean(mi)

def svm_cv_score(X, y, cv=5):
    """
    SVM交叉验证评分，衡量波段集区分力
    X, y: same as above
    """
    svm = SVC(kernel='linear')
    try:
        scores = cross_val_score(svm, X, y, cv=cv)
        return scores.mean()
    except Exception:
        return 0.0

def fisher_criterion(X, y):
    """
    Fisher判别标准
    """
    labels = set(y)
    means = {l: X[np.array(y) == l].mean(axis=0) for l in labels}
    n = X.shape[0]
    overall_mean = X.mean(axis=0)
    S_b = sum([(np.sum(np.array(y) == l)) * np.square(means[l] - overall_mean) for l in labels])
    S_w = sum([((X[np.array(y) == l] - means[l]) ** 2).sum(axis=0) for l in labels])
    fisher_score = S_b.sum() / (S_w.sum() + 1e-12)
    return fisher_score

def composite_score(X, y, weights=(0.5, 0.5, 0.0)):
    """
    综合评价函数：加权和，weights = (互信息, SVM, Fisher)
    """
    mi = mutual_info_score(X, y)
    svm = svm_cv_score(X, y)
    fisher = fisher_criterion(X, y)
    w_mi, w_svm, w_fisher = weights
    score = w_mi * mi + w_svm * svm + w_fisher * fisher
    return score, {"mutual_info": mi, "svm_cv": svm, "fisher": fisher}