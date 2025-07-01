import numpy as np
from scipy.spatial.distance import pdist


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


def greedy_band_selection(
        X, y,
        band_pool=None,
        n_select=7,
        n_trials=30,
        disturb_prob=0.3, # 每轮有概率随机替换掉一个已选band
        random_state=None
    ):
    """
    改进：每轮贪心基础上，带入一定概率的随机扰动（扰动后再局部贪心探索），多轮重复，保留最佳
    - n_trials: 总共运行多少轮
    - disturb_prob: 每轮扰动的概率（扰动已选波段，防止陷入局部最优）
    - random_state: 控制随机性
    """
    rng = np.random.default_rng(random_state)
    if band_pool is None:
        band_pool = np.arange(X.shape[1])
    band_pool = np.array(band_pool)
    best_combo = None
    best_score = -np.inf

    for trial in range(n_trials):
        selected = []
        local_band_pool = band_pool.copy()

        for i in range(n_select):
            # 有概率进行扰动：随便pick一个未被选的band
            if rng.random() < disturb_prob and len(local_band_pool) > len(selected):
                # 从未被选的band中随机pick一个
                choices = [b for b in local_band_pool if b not in selected]
                best_band = rng.choice(choices)
                cand = selected + [best_band]
            else:
                # 常规贪心
                best_band = None
                best_band_score = -np.inf
                for band in local_band_pool:
                    if band in selected:
                        continue
                    cand = selected + [band]
                    X_sub = X[:, cand]
                    mask = ~np.any(np.isnan(X_sub), axis=1)
                    if np.sum(mask) < 2 or len(np.unique(y[mask])) < 2:
                        continue
                    score = group_discriminability(X_sub[mask], y[mask])
                    if score > best_band_score:
                        best_band_score = score
                        best_band = band
                if best_band is None:
                    break
                cand = selected + [best_band]

            X_sub = X[:, cand]
            mask = ~np.any(np.isnan(X_sub), axis=1)
            if np.sum(mask) < 2 or len(np.unique(y[mask])) < 2:
                continue
            selected.append(best_band)

        # 组合完成后，评估
        if len(selected) < n_select: continue
        X_sub = X[:, selected]
        mask = ~np.any(np.isnan(X_sub), axis=1)
        score = group_discriminability(X_sub[mask], y[mask])
        if score > best_score:
            best_score = score
            best_combo = selected.copy()
    return best_combo, best_score


def sffs_band_selection(
        X, y,
        band_pool=None,
        n_select=7,
        score_func=None,
        n_trials=20,
        disturb_prob=0.2,
        random_state=None
    ):
    """
    随机浮动 SFFS：每次迭代有概率踢掉/增添一个随机band，在普通sffs基础上增加扰动和多轮尝试
    """
    rng = np.random.default_rng(random_state)
    if score_func is None:
        score_func = group_discriminability
    if band_pool is None:
        band_pool = np.arange(X.shape[1])
    band_pool = np.array(band_pool)

    best_score = -np.inf
    best_combo = None
    for trial in range(n_trials):
        selected = []
        local_band_pool = band_pool.copy()

        while len(selected) < n_select:
            # 有概率扰动：加一个随机band
            do_disturb = (rng.random() < disturb_prob and len(selected) > 0)
            if do_disturb:
                choices = [b for b in local_band_pool if b not in selected]
                if not choices:
                    break
                band = rng.choice(choices)
                selected.append(band)
                selected = list(set(selected))[:n_select]
            else:
                # 标准SFFS向前
                best_add, best_add_score = None, -np.inf
                for band in local_band_pool:
                    if band in selected:
                        continue
                    cand = selected + [band]
                    X_sub = X[:, cand]
                    mask = ~np.any(np.isnan(X_sub), axis=1)
                    if np.sum(mask) < 2 or len(np.unique(y[mask])) < 2:
                        continue
                    score = score_func(X_sub[mask], y[mask])
                    if score > best_add_score:
                        best_add_score = score
                        best_add = band
                if best_add is not None:
                    selected.append(best_add)
                else:
                    break

            # 浮动后退，照常
            float_flag = True
            while float_flag and len(selected) > 1:
                best_remove, best_remove_score = None, -np.inf
                for b in selected:
                    cand = [x for x in selected if x != b]
                    X_sub = X[:, cand]
                    mask = ~np.any(np.isnan(X_sub), axis=1)
                    if np.sum(mask) < 2 or len(np.unique(y[mask])) < 2:
                        continue
                    score = score_func(X_sub[mask], y[mask])
                    if score > best_remove_score:
                        best_remove = b
                        best_remove_score = score
                if best_remove_score > score_func(X[:, selected], y):
                    selected.remove(best_remove)
                else:
                    float_flag = False

        if len(selected) < n_select: continue
        X_sub = X[:, selected]
        mask = ~np.any(np.isnan(X_sub), axis=1)
        score = score_func(X_sub[mask], y[mask])
        if score > best_score:
            best_score = score
            best_combo = selected.copy()
    return best_combo, best_score