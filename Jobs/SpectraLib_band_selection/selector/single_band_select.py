import pandas as pd
import numpy as np


def variance_band_select(variances, valid_indices, topk):
    """
    只对外部传入的有效波段索引进行排序、选择topk
    """
    valid_variances = variances[valid_indices]
    sort_idx = np.argsort(valid_variances)[::-1][:topk]
    top_indices = valid_indices[sort_idx]
    return top_indices, valid_variances[sort_idx]

def range_band_select(ranges, valid_indices, topk):
    """
    只对外部传入的有效波段索引进行排序、选择topk
    """
    valid_ranges = ranges[valid_indices]
    sort_idx = np.argsort(valid_ranges)[::-1][:topk]
    top_indices = valid_indices[sort_idx]
    return top_indices, valid_ranges[sort_idx]


def fisher_band_select(X, y, topk):
    # 单样本/类别情况，Fisher等价range，这里还是实现为可扩展多样本分组
    if y is None:
        # 每行一个类别且只有一行，等价于极差
        scores = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
    else:
        unique_cls = np.unique(y)
        n_band = X.shape[1]
        scores = []
        for band in range(n_band):
            band_vals = X[:, band]
            mask = ~np.isnan(band_vals)
            global_mean = np.nanmean(band_vals)
            S_b, S_w = 0, 0
            for cls in unique_cls:
                cls_vals = band_vals[(y==cls) & mask]
                if len(cls_vals) == 0:
                    continue
                cls_mean = np.nanmean(cls_vals)
                S_b += (cls_mean - global_mean) ** 2 * len(cls_vals)
                S_w += np.nansum((cls_vals - cls_mean) ** 2)
            score = S_b / (S_w + 1e-8)
            scores.append(score)
        scores = np.array(scores)
    rank = np.argsort(scores)[::-1]
    return rank[:topk], scores[rank[:topk]]

def report_result_simple(selected_bands_dict, band_names=None):
    df_report = []
    for method, bands in selected_bands_dict.items():
        band_idxs, band_scores = bands
        if band_names is not None:
            bands_disp = [band_names[i] for i in band_idxs]
        else:
            bands_disp = [f"Band_{i}" for i in band_idxs]
        df_report.append({
            "Method": method,
            "BandIdx": band_idxs.tolist(),
            "BandNames": bands_disp,
            "BandScores": band_scores,
        })
    return pd.DataFrame(df_report)