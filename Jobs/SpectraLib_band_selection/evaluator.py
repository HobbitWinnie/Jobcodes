import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

def mutual_info_score(X, y):
    """
    对于每个特征，仅在非NaN类别评估互信息，最后取均值
    """
    X = np.asarray(X)
    y = np.asarray(y)
    mi_vals = []
    for j in range(X.shape[1]):
        col_mask = ~np.isnan(X[:, j])
        X_col = X[col_mask, j].reshape(-1, 1)
        y_col = y[col_mask]
        if X_col.shape[0] < 2 or len(np.unique(y_col)) < 2:
            mi_vals.append(0.0)
        else:
            mi = mutual_info_classif(X_col, y_col, discrete_features=False, random_state=0)
            mi_vals.append(mi[0])
    return float(np.mean(mi_vals))

def svm_cv_score(X, y):
    """
    SVM用NaN填充为0代替（可选择其他值），因为类别均值矩阵没法做掩码训练
    """
    y = np.asarray(y)
    X_filled = np.nan_to_num(X, nan=0.0)  # 也可用其他填充值
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        return 0.0
    try:
        clf = SVC(kernel='linear')
        clf.fit(X_filled, y)
        score = clf.score(X_filled, y)
        return float(score)
    except Exception:
        return 0.0

def fisher_criterion(X, y):
    """
    Fisher：每个特征只在非NaN类别评估
    """
    X = np.asarray(X)
    y = np.asarray(y)
    scores = []
    for j in range(X.shape[1]):
        col_mask = ~np.isnan(X[:, j])
        X_col = X[col_mask, j]
        y_col = y[col_mask]
        if X_col.shape[0] < 2 or len(np.unique(y_col)) < 2:
            continue
        classes = np.unique(y_col)
        means = [X_col[y_col == c].mean() for c in classes]
        counts = [np.sum(y_col == c) for c in classes]
        overall_mean = X_col.mean()
        S_b = sum([counts[k] * (means[k] - overall_mean) ** 2 for k in range(len(classes))])
        S_w = sum([((X_col[y_col == c] - means[k]) ** 2).sum() for k, c in enumerate(classes)])
        if S_w > 0:
            scores.append(S_b / (S_w + 1e-12))
    return float(np.mean(scores)) if scores else 0.0

def composite_score(X, y, weights=(0.5, 0.5, 0.0)):
    mi    = mutual_info_score(X, y)
    svm   = svm_cv_score(X, y)
    fisher= fisher_criterion(X, y)
    w_mi, w_svm, w_fisher = weights
    score = w_mi * mi + w_svm * svm + w_fisher * fisher
    return score, {"mutual_info": mi, "svm_cv": svm, "fisher": fisher}