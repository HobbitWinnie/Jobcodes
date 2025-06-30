import numpy as np
from scipy.spatial.distance import pdist
import itertools

def group_discriminability(X, y):
    """给定样本X（n, k），y，返回组间中心均值距离/组内方差之比"""
    cls_labels = np.unique(y)
    centers = []
    for cls in cls_labels:
        vals = X[y == cls]
        if len(vals) == 0:
            continue
        center = np.mean(vals, axis=0)
        centers.append(center)
    centers = np.array(centers)
    if len(centers) < 2:
        return 0.0  # 只有一个类无意义
    inter_dist = np.mean(pdist(centers, metric='euclidean'))
    # 组内（类内）总方差均值
    intra_var = 0
    count = 0
    for cls, center in zip(cls_labels, centers):
        vals = X[y == cls]
        if len(vals) > 1:
            var = np.sum((vals - center) ** 2)
            intra_var += var
            count += vals.shape[0]
    intra_var_mean = (intra_var / count) if count > 0 else 1e-8  # 防0
    return inter_dist / (intra_var_mean**0.5 + 1e-8)

def greedy_band_combination(X, y, band_pool=None, n_select=7):
    """
    贪心递进法：每次增加1个band，组合区分力最大化
    X: (n, b) 原始数据
    y: (n,) 标签
    band_pool: 候选波段索引（如先用单band方法筛，建议>n_select即可）
    n_select: 想选多少个波段组成组合
    return: 最佳组合的band索引列表、最终组合区分力
    """
    if band_pool is None:
        band_pool = np.arange(X.shape[1])
    else:
        band_pool = np.array(band_pool)
    
    selected = []
    for i in range(n_select):
        best_band = None
        best_score = -np.inf
        for band in band_pool:
            if band in selected:
                continue
            cand = selected + [band]
            X_sub = X[:, cand]
            # 剔除有nan的样本，保证组合特征不nan
            mask = ~np.any(np.isnan(X_sub), axis=1)
            if np.sum(mask) < 2 or len(np.unique(y[mask])) < 2:
                continue
            score = group_discriminability(X_sub[mask], y[mask])
            if score > best_score:
                best_score = score
                best_band = band
        if best_band is not None:
            selected.append(best_band)
        else:
            break  # 没法再加了
    return selected, best_score