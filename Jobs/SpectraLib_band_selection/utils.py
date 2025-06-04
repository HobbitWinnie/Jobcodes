import numpy as np
import itertools

def get_combinations(n_bands, comb_dim):
    """
    生成所有波段组合
    """
    return list(itertools.combinations(range(n_bands), comb_dim))

def get_label_indices(labels, target_list):
    """
    获得目标标签在样本集中的下标
    """
    return [i for i, l in enumerate(labels) if l in target_list]